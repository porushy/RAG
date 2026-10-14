"""Answer questions about the Economic Survey chapters with a cited, grounded LLM answer.

Pipeline (Phase 7 of pdf_study_assistant_part3.ipynb, which explains every step below):

    1. Rebuild the retriever from Phases 4-6 (pdf_study_assistant_2.py).
    2. Format the retrieved chunks as numbered sources headed by their pages.
    3. Ask Gemini, through an LCEL chain, to answer only from those sources, citing pages,
       or to say the document does not contain the answer.
    4. Retry on HTTP 429/503 with exponential backoff; stop at once on a daily quota.
    5. Check that every cited page was among the pages supplied.

All LLM calls live in this file, so the model or provider can change without touching
the retrieval code. Needs GOOGLE_API_KEY in the project's .env.

Usage (from the project root or from notebooks/):

    python notebooks/pdf_study_assistant_3.py --question TEXT [--question TEXT ...] [--model ID]
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

import pdf_study_assistant_1 as cleaning
import pdf_study_assistant_2 as pipeline

logger = logging.getLogger(__name__)

# gemini-3.8-flash allows 20 free requests a day; 3.7-flash has its own quota (checked 2026-10-13).
GEMINI_MODEL = "gemini-3.7-flash"
RETRYABLE_STATUS_CODES = {429, 503}  # too many requests; server overloaded
MAX_WAIT_SECONDS = 60  # a longer suggested wait means a daily quota: retrying cannot help
NOT_IN_DOCUMENT = "The document does not contain this information."
CITATION_PATTERN = re.compile(r"\[pp?\. (\d+)(?:-(\d+))?\]")  # [p. 22] or [pp. 55-56]

SYSTEM_INSTRUCTIONS = f"""You are a study assistant for chapters of India's Economic Survey.
Answer the question using ONLY the numbered sources in the context.
- Do not use any knowledge from outside the context, even if you know the answer.
- After each fact, cite the page it came from in square brackets, exactly as written in the
  source header, for example [p. 55] or [pp. 55-56].
- If the context does not contain the answer, reply with exactly this sentence and nothing
  else: "{NOT_IN_DOCUMENT}"
- Keep the answer short: a few sentences."""


# ------------------------------------------------------------------------------- LLM


def load_api_key() -> None:
    """Load GOOGLE_API_KEY from the project's .env into the environment, or fail clearly."""
    load_dotenv(cleaning.PROJECT_ROOT / ".env")
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise RuntimeError(f"GOOGLE_API_KEY is missing. Copy .env.example to .env in {cleaning.PROJECT_ROOT}.")


def make_llm(model: str = GEMINI_MODEL) -> ChatGoogleGenerativeAI:
    """Gemini chat model: deterministic, no reasoning pass, one attempt (we retry ourselves)."""
    return ChatGoogleGenerativeAI(model=model, temperature=0, thinking_budget=0, max_retries=1)


def http_status_of(error: BaseException | None) -> int | None:
    """The HTTP status code carried by an error or by any error it was raised from."""
    while error is not None:
        code = getattr(error, "code", None)
        if isinstance(code, int):
            return code
        error = error.__cause__
    return None


def suggested_wait_of(error: BaseException | None) -> float | None:
    """Seconds Google asks us to wait before retrying (its RetryInfo entry), else None."""
    while error is not None:
        details = getattr(error, "details", None)
        if isinstance(details, dict):
            for entry in details.get("error", {}).get("details", []):
                if entry.get("@type", "").endswith("RetryInfo"):
                    return float(entry["retryDelay"].rstrip("s"))
        error = error.__cause__
    return None


def invoke_with_backoff(runnable: Runnable, chain_input, max_attempts: int = 6, initial_delay_seconds: float = 2.0):
    """Invoke a Runnable, retrying on HTTP 429/503 with exponentially growing, jittered waits."""
    for attempt in range(1, max_attempts + 1):
        try:
            return runnable.invoke(chain_input)
        except Exception as error:
            status = http_status_of(error)
            if status not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
                raise
            suggested = suggested_wait_of(error)
            if suggested is not None and suggested > MAX_WAIT_SECONDS:
                raise RuntimeError(
                    f"HTTP {status}: Google asks us to wait {suggested / 3600:.1f} hours. The daily "
                    "free-tier quota for this model is used up; try again after it resets."
                ) from error
            delay = max(initial_delay_seconds * 2 ** (attempt - 1), suggested or 0) + random.uniform(0, 1)
            logger.info("HTTP %s on attempt %d; waiting %.1f s before retrying", status, attempt, delay)
            time.sleep(delay)


# --------------------------------------------------------------------------- the chain


def page_reference(page_numbers: list[int]) -> str:
    """Citation label for a chunk's pages: [22] -> 'p. 22', [55, 56] -> 'pp. 55-56'."""
    if len(page_numbers) == 1:
        return f"p. {page_numbers[0]}"
    return f"pp. {page_numbers[0]}-{page_numbers[-1]}"


def format_docs(chunks) -> str:
    """Join retrieved chunks into one context string, each headed by its source number and pages."""
    return "\n\n".join(
        f"[Source {number} | {page_reference(chunk.metadata['page_numbers'])}]\n{chunk.page_content}"
        for number, chunk in enumerate(chunks, start=1)
    )


def make_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_INSTRUCTIONS), ("human", "Context:\n\n{context}\n\nQuestion: {question}")]
    )


def build_rag_chain(retriever: Runnable, llm: Runnable) -> Runnable:
    """question -> answer text (the brief's LCEL chain)."""
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | make_prompt()
        | llm
        | StrOutputParser()
    )


def build_rag_chain_with_sources(retriever: Runnable, llm: Runnable) -> Runnable:
    """question -> {"source_documents", "question", "answer"}; the retriever runs once."""
    answer_from_sources = (
        RunnablePassthrough.assign(context=lambda inputs: format_docs(inputs["source_documents"]))
        | make_prompt()
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(source_documents=retriever, question=RunnablePassthrough()).assign(
        answer=answer_from_sources
    )


def unsupported_citations(answer: str, source_chunks) -> tuple[set[int], set[int]]:
    """Pages cited in the answer, and those among them that no source chunk covers."""
    supplied = {page for chunk in source_chunks for page in chunk.metadata["page_numbers"]}
    cited = set()
    for first, last in CITATION_PATTERN.findall(answer):
        cited.update(range(int(first), int(last or first) + 1))
    return cited, cited - supplied


# ------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--question", action="append", required=True, help="question to answer (repeatable)")
    parser.add_argument("--model", default=GEMINI_MODEL, help="Gemini model ID")
    parser.add_argument("--k", type=int, default=pipeline.TOP_K, help="chunks retrieved per question")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")  # quiet HTTP and FAISS loader logs
    logger.setLevel(logging.INFO)

    load_api_key()
    chapters = cleaning.load_documents(cleaning.DEFAULT_OUTPUT_DIR / cleaning.CHAPTERS_FILENAME)
    chunks = pipeline.chunk_chapters(chapters)
    embedding_model = pipeline.load_embedding_model()
    vector_store, _ = pipeline.build_or_load_index(chunks, embedding_model, pipeline.DEFAULT_INDEX_DIR)
    pipeline.verify_index(pipeline.DEFAULT_INDEX_DIR, chunks, embedding_model)
    chain = build_rag_chain_with_sources(pipeline.make_retriever(vector_store, args.k), make_llm(args.model))

    for question in args.question:
        result = invoke_with_backoff(chain, question)
        answer, sources = result["answer"], result["source_documents"]
        cited, unsupported = unsupported_citations(answer, sources)
        logger.info("\nQUESTION: %s\nANSWER  : %s", question, answer)
        for number, chunk in enumerate(sources, start=1):
            logger.info("  source %d  %-10s %s ...", number, page_reference(chunk.metadata["page_numbers"]),
                        chunk.page_content[:80].replace("\n", " "))
        if answer.strip() == NOT_IN_DOCUMENT:
            logger.info("CHECK   : the model says the document does not contain the answer")
        else:
            logger.info("CHECK   : cited pages %s; all among the sources: %s%s", sorted(cited), not unsupported,
                        f" (unsupported: {sorted(unsupported)})" if unsupported else "")


if __name__ == "__main__":
    main()
