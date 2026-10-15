"""Answer questions about the Economic Survey chapters with a cited, grounded LLM answer.

Pipeline (Phase 7 of pdf_study_assistant_part3.ipynb, which explains every step; the
functions themselves live in rag_utils.py):

    1. Rebuild the retriever from Phases 4-6.
    2. Format the retrieved chunks as numbered sources headed by their pages.
    3. Ask Gemini, through an LCEL chain, to answer only from those sources, citing pages,
       or to say the document does not contain the answer.
    4. Retry on HTTP 429/503 with exponential backoff; stop at once on a daily quota.
    5. Check that every cited page was among the pages supplied.

All LLM calls happen here, so the model or provider can change without touching the
retrieval code. Needs GOOGLE_API_KEY in the project's .env.

Usage (from the project root or from notebooks/):

    python notebooks/pdf_study_assistant_3.py --question TEXT [--question TEXT ...] [--model ID]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # rag_utils.py is in the project root

# Some names are not used below but re-exported, for code that imported them from this module.
from rag_utils import (  # noqa: E402
    GEMINI_MODEL,
    NOT_IN_DOCUMENT,
    TOP_K,
    build_rag_chain,
    build_rag_chain_with_sources,
    cited_pages_of,
    format_docs,
    invoke_with_backoff,
    load_api_key,
    load_retriever,
    page_reference,
    unsupported_citations,
    uses_required_form,
)
from rag_utils import make_gemini_llm as make_llm  # noqa: E402

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--question", action="append", required=True, help="question to answer (repeatable)")
    parser.add_argument("--model", default=GEMINI_MODEL, help="Gemini model ID")
    parser.add_argument("--k", type=int, default=TOP_K, help="chunks retrieved per question")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")  # quiet HTTP and FAISS loader logs
    for name in (__name__, "rag_utils"):
        logging.getLogger(name).setLevel(logging.INFO)

    load_api_key("GOOGLE_API_KEY")
    stack = load_retriever(k=args.k)
    chain = build_rag_chain_with_sources(stack.retriever, make_llm(args.model))

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
        elif not cited:
            logger.info("CHECK   : WARNING: the answer cites no page at all")
        else:
            logger.info("CHECK   : cited pages %s; all among the sources: %s%s; exact [p. N] form: %s",
                        sorted(cited), not unsupported,
                        f" (unsupported: {sorted(unsupported)})" if unsupported else "", uses_required_form(answer))


if __name__ == "__main__":
    main()
