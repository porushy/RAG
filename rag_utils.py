"""Reusable functions for the PDF study assistant: a RAG pipeline over five chapters of
India's Economic Survey.

The notebooks (notebooks/*.ipynb) explain and check every step; this module holds the working
code, so scripts and notebooks share one implementation. Sections follow the project's phases:

    Phases 2-3  load the PDF, clean each page, join chapters, save JSON Lines
    Phase 4     split chapters into chunks labelled with their page ranges
    Phase 5     embed chunks (all-MiniLM-L6-v2, normalised) into a FAISS inner-product index
    Phase 6     retrieve the top-k chunks with cosine scores
    Phase 7     answer with an LLM (Gemini, or Groq with a fallback model) through LCEL,
                with exponential backoff and citation checks
    Phase 8     score retrieval (hit@k) and answers (key facts) against an evaluation set

Typical use:

    import rag_utils as rag
    rag.prepare_corpus()                       # once: PDF -> data/processed/*.jsonl
    stack = rag.load_retriever()               # chunks, embeddings, FAISS index, retriever
    chain = rag.build_rag_chain_with_sources(stack.retriever, rag.make_gemini_llm())
    result = rag.invoke_with_backoff(chain, "How many GCCs does India host?")
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "survey_chapters.pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PAGES_FILENAME = "cleaned_documents.jsonl"
CHAPTERS_FILENAME = "cleaned_chapters.jsonl"
DEFAULT_CHAPTERS_PATH = DEFAULT_OUTPUT_DIR / CHAPTERS_FILENAME
DEFAULT_INDEX_DIR = PROJECT_ROOT / "faiss_index"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "eval" / "eval_questions.json"

# ----------------------------------------------------------- Phases 2-3: cleaning rules

FIRST_PAGE_IN_ORIGINAL_PDF = 276  # first page of the slice in the full Survey (download_data.py)
BOILERPLATE_THRESHOLD = 10  # a first line repeated on this many pages is a running head
HEAD_LINES_TO_CHECK = 3  # running heads are only removed near the top of a page
MINIMUM_LETTER_RATIO = 0.20  # lines below this are flattened chart data

# The notice printed on the spacer pages between chapters; only these pages are dropped.
BLANK_PAGE_PATTERN = re.compile(r"\s*This page has been left blank\.?\s*", re.IGNORECASE)
# A footnote line: its number, whitespace, then the note ("92  Department of Fisheries.").
FOOTNOTE_LINE = re.compile(r"^\s*(\d{1,3})\s+\S")
# Chapter-opening pages; footnote numbering restarts there.
CHAPTER_START = re.compile(r"^\s*CHAPTER", re.MULTILINE)
# Footnote-block safeguards: blocks sit at the page foot (longest real block: 19 lines),
# and no numbered body paragraph ever follows them.
MAX_LINES_FROM_BOTTOM = 25
PARAGRAPH_NUMBER_LINE = re.compile(r"^\s*(?:6|7|8|9|10)\.\d{1,3}\.?\s")
APPARATUS_PATTERN = re.compile(r"^\s*(?:Sources?|Notes?)\s*:", re.IGNORECASE)
HYPHEN_LINEBREAK = re.compile(r"([A-Za-z])-[ \t]*\n[ \t]*([A-Za-z])")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
FOOTNOTE_AFTER_WORD = re.compile(r"(?<=[a-z])\.\d{1,2}(?=\s+[A-Z])")
# Four digits, so that paragraph markers such as 8.44 and 10.98 cannot match.
FOOTNOTE_AFTER_YEAR = re.compile(r"(?<=\d{4})\.\d{1,2}(?=\s+[A-Z])")
# The document's own paragraph numbers: chapters 6-10, followed by text.
PARAGRAPH_MARKER = re.compile(r"(?m)^[ \t]*((?:6|7|8|9|10)\.\d{1,3})\.?[ \t]+(?=[A-Za-z(])")
# A full stop followed by spaces ends a sentence; decimals (5.6) and paragraph numbers
# (8.44) have no space after the dot, so they never match.
SENTENCE_BREAK = re.compile(r"\. +")
# A finished sentence, allowing closing quotes/brackets and a leftover footnote number.
SENTENCE_END = re.compile(r"[.?!:;][”’\")]*\d{0,3}$")
# Ways a page can open a new block instead of continuing a paragraph.
NEW_BLOCK_START = re.compile(
    r"^(?:"
    r"(?:6|7|8|9|10)\.\d{1,3}\s"
    r"|(?:Chart\w*|Box|Table|Figure)\s+[IVX]+\.\d+\s*:"
    r"|Source\s*:"
    r"|CHAPTER"
    r"|•"
    r"|[A-Z][A-Z ,&:’'\-]{9,}(?:\s|$)"
    r")"
)

# ------------------------------------------------- Phases 4-6: chunks, index, retrieval

CHUNK_SIZE = 1000  # maximum characters per chunk
CHUNK_OVERLAP = 200  # at most this many characters repeated from the previous chunk
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Unit vectors: inner product = cosine similarity. LangChain's default is Euclidean, and
# the saved index does not record the strategy, so it is passed on build and on load.
DISTANCE_STRATEGY = DistanceStrategy.MAX_INNER_PRODUCT
TOP_K = 5  # chunks returned per question

# --------------------------------------------------------------------- Phase 7: the LLM

# gemini-3.8-flash allows 20 free requests a day; 3.7-flash has its own quota (checked 2026-10-13).
GEMINI_MODEL = "gemini-3.7-flash"
# Groq (checked 2026-10-15): 1,000 requests/day and 8,000 tokens/minute per model on the free tier.
GROQ_PRIMARY_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODEL = "qwen/qwen3.8-27b"
RETRYABLE_STATUS_CODES = {429, 503}  # too many requests; server overloaded
MAX_WAIT_SECONDS = 60  # a longer suggested wait means a daily quota: retrying cannot help
STATUS_IN_MESSAGE = re.compile(r"^\[(\d{3})\]")  # errors raised as "[429] Too Many Requests ..."
NOT_IN_DOCUMENT = "The document does not contain this information."
# Citations are read tolerantly (models write "[Source 1 | p. 55]", "【p. 129】", odd dashes);
# whether they use the exact requested form is reported separately.
BRACKETED = re.compile(r"[\[【]([^\]】]*)[\]】]")
PAGE_REFERENCE = re.compile(r"pp?\.\s*(\d+)(?:\s*[-‐‑–—]\s*(\d+))?")
REQUIRED_FORM = re.compile(r"\[pp?\. \d+(?:-\d+)?\]")

SYSTEM_INSTRUCTIONS = f"""You are a study assistant for chapters of India's Economic Survey.
Answer the question using ONLY the numbered sources in the context.
- Do not use any knowledge from outside the context, even if you know the answer.
- After each fact, cite the page it came from in square brackets, exactly as written in the
  source header, for example [p. 55] or [pp. 55-56].
- If the context does not contain the answer, reply with exactly this sentence and nothing
  else: "{NOT_IN_DOCUMENT}"
- Keep the answer short: a few sentences."""


# --------------------------------------------------------------------- Phase 2: loading


def load_pages(pdf_path: Path) -> list[Document]:
    """Read the PDF into one Document per page."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"{pdf_path} not found. Run `python download_data.py` from the project root first."
        )
    pages = PyPDFLoader(str(pdf_path)).load()
    logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)
    return pages


# ---------------------------------------------------------- Phase 3: corpus-wide passes


def discover_running_heads(pages: list[Document], threshold: int = BOILERPLATE_THRESHOLD) -> set[str]:
    """Return first lines that repeat across many pages (headers, not prose)."""
    first_lines = []
    for page in pages:
        non_empty = [line.strip() for line in page.page_content.splitlines() if line.strip()]
        if non_empty:
            first_lines.append(non_empty[0])
    counts = Counter(first_lines)
    return {line for line, count in counts.items() if count >= threshold}


def footnote_number(line: str) -> int | None:
    """The number a footnote-shaped line starts with, otherwise None."""
    match = FOOTNOTE_LINE.match(line)
    return int(match.group(1)) if match else None


def is_near_bottom(lines: list[str], line_index: int) -> bool:
    """True if the line is within MAX_LINES_FROM_BOTTOM lines of the end of the page."""
    return len(lines) - line_index <= MAX_LINES_FROM_BOTTOM


def has_clean_tail(lines: list[str], line_index: int) -> bool:
    """True if no paragraph-numbered body line comes after this line."""
    return not any(PARAGRAPH_NUMBER_LINE.match(line) for line in lines[line_index + 1:])


def find_footnote_blocks(pages: list[Document]) -> dict[int, tuple[int, int, int]]:
    """Locate each page's footnote block by following the document's footnote numbering.

    A block must start with exactly the next expected footnote number (numbering
    restarts at each chapter), near the bottom of the page, with no numbered body
    paragraph after it; body lines or chart values that merely start with a number
    are therefore not mistaken for footnotes.

    Returns {page_index: (first_line_index, first_number, last_number)}.
    """
    blocks = {}
    expected_number = 1
    for page_index, page in enumerate(pages):
        lines = page.page_content.splitlines()
        if CHAPTER_START.search("\n".join(lines[:4])):
            expected_number = 1
        candidates = [
            index
            for index, line in enumerate(lines)
            if footnote_number(line) == expected_number
            and is_near_bottom(lines, index)
            and has_clean_tail(lines, index)
        ]
        if not candidates:
            continue
        first_line = candidates[-1]
        last_number = expected_number
        for line in lines[first_line + 1:]:
            if footnote_number(line) == last_number + 1:
                last_number += 1
        blocks[page_index] = (first_line, expected_number, last_number)
        expected_number = last_number + 1
    return blocks


# ---------------------------------------------------------- Phase 3: cleaning each page


def letter_ratio(line: str) -> float:
    """Fraction of a line's characters that are alphabetic letters."""
    if not line:
        return 0.0
    return sum(character.isalpha() for character in line) / len(line)


def remove_footnote_block(text: str, block: tuple[int, int, int] | None) -> str:
    """Cut a page's text off where its footnote block begins."""
    if block is None:
        return text
    return "\n".join(text.splitlines()[: block[0]])


def remove_running_heads(text: str, heads: set[str], lines_to_check: int = HEAD_LINES_TO_CHECK) -> str:
    """Delete running-head lines, but only near the top of the page.

    Some heads ('Environment and Climate Change') also occur in body prose;
    position is what tells them apart.
    """
    return "\n".join(
        line
        for position, line in enumerate(text.splitlines())
        if not (position < lines_to_check and line.strip() in heads)
    )


def remove_junk_lines(text: str, minimum_letter_ratio: float = MINIMUM_LETTER_RATIO) -> str:
    """Delete lines that are almost entirely numbers (flattened chart data)."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.strip() or letter_ratio(line.strip()) >= minimum_letter_ratio
    )


def remove_apparatus(text: str) -> str:
    """Delete chart 'Source:' and 'Note:' lines."""
    return "\n".join(line for line in text.splitlines() if not APPARATUS_PATTERN.match(line))


def join_hyphenated_linebreaks(text: str) -> str:
    """Rejoin compounds split across lines, keeping the hyphen.

    Every line-end hyphen in this corpus is a real compound ('MSP-based'), so
    deleting the hyphen would corrupt words.
    """
    return HYPHEN_LINEBREAK.sub(r"\1-\2", text)


def strip_urls(text: str) -> str:
    """Remove web addresses."""
    return URL_PATTERN.sub(" ", text)


def repair_footnote_markers(text: str) -> str:
    """Detach footnote numbers fused to a word or year ('livelihoods.2 A')."""
    text = FOOTNOTE_AFTER_WORD.sub(".", text)
    return FOOTNOTE_AFTER_YEAR.sub(".", text)


def mark_paragraphs(text: str) -> str:
    """Insert a blank line before each of the document's own paragraph numbers."""
    return PARAGRAPH_MARKER.sub(r"\n\n\1 ", text)


def collapse_whitespace(text: str) -> str:
    """Turn line-wrap newlines into spaces while preserving paragraph breaks."""
    paragraphs = (re.sub(r"[ \t]+", " ", block.replace("\n", " ")).strip() for block in text.split("\n\n"))
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def break_sentences(text: str) -> str:
    """Start a new line after every sentence-ending full stop, so chunking can cut there."""
    return SENTENCE_BREAK.sub(".\n", text)


def clean_page(text: str, heads: set[str], footnote_block: tuple[int, int, int] | None = None) -> str:
    """Run cleaning Steps 0-9. The order matters."""
    text = remove_footnote_block(text, footnote_block)  # 0  needs the original line layout
    text = remove_running_heads(text, heads)  # 1
    text = remove_junk_lines(text)  # 2  must precede 7: removes chart values shaped like markers
    text = remove_apparatus(text)  # 3
    text = join_hyphenated_linebreaks(text)  # 4  must precede 5: reunites split URLs
    text = strip_urls(text)  # 5
    text = repair_footnote_markers(text)  # 6
    text = mark_paragraphs(text)  # 7
    text = collapse_whitespace(text)  # 8
    text = break_sentences(text)  # 9  must follow 8, which turns line breaks into spaces
    return text


def is_blank_spacer(text: str) -> bool:
    """True only when the whole page is the 'This page has been left blank' notice."""
    return BLANK_PAGE_PATTERN.fullmatch(text) is not None


def clean_documents(
    pages: list[Document], heads: set[str], footnote_blocks: dict[int, tuple[int, int, int]]
) -> tuple[list[Document], list[int]]:
    """Clean every page; return the kept Documents and the indices of dropped spacer pages."""
    cleaned, dropped = [], []
    for page_index, page in enumerate(pages):
        text = clean_page(page.page_content, heads, footnote_blocks.get(page_index))
        if is_blank_spacer(text):
            dropped.append(page_index)
            continue
        metadata = {
            **page.metadata,
            "page_number": page_index + 1,
            "original_page": page_index + FIRST_PAGE_IN_ORIGINAL_PDF,
        }
        cleaned.append(Document(page_content=text, metadata=metadata))
    return cleaned, dropped


# -------------------------------------------------------- Phase 3: joining across pages


def continues_previous_page(previous_text: str, current_text: str) -> bool:
    """True if current_text carries on the paragraph that previous_text broke off."""
    ends_mid_sentence = SENTENCE_END.search(previous_text.rstrip()) is None
    opens_new_block = NEW_BLOCK_START.match(current_text) is not None
    return ends_mid_sentence and not opens_new_block


def flag_continuations(documents: list[Document]) -> int:
    """Set metadata['continues_previous_page'] on every page; return how many are True."""
    documents[0].metadata["continues_previous_page"] = False
    for previous_doc, current_doc in zip(documents, documents[1:]):
        adjacent = current_doc.metadata["page_number"] == previous_doc.metadata["page_number"] + 1
        current_doc.metadata["continues_previous_page"] = adjacent and continues_previous_page(
            previous_doc.page_content, current_doc.page_content
        )
    return sum(doc.metadata["continues_previous_page"] for doc in documents)


def build_chapter_documents(documents: list[Document]) -> list[Document]:
    """Join each chapter's pages into one text, recording where every page starts.

    Continuing pages are joined with a space (same paragraph), others with a blank line.
    """
    chapters = []
    for doc in documents:
        if doc.page_content.startswith("CHAPTER") or not chapters:
            chapters.append({"text": "", "pages": []})
        chapter = chapters[-1]
        if chapter["text"]:
            chapter["text"] += " " if doc.metadata["continues_previous_page"] else "\n\n"
        chapter["pages"].append((len(chapter["text"]), doc.metadata))
        chapter["text"] += doc.page_content

    chapter_documents = []
    for chapter in chapters:
        page_metadata = [metadata for _, metadata in chapter["pages"]]
        chapter_documents.append(
            Document(
                page_content=chapter["text"],
                metadata={
                    "source": page_metadata[0]["source"],
                    "page_starts": [start for start, _ in chapter["pages"]],
                    "page_numbers": [metadata["page_number"] for metadata in page_metadata],
                    "page_labels": [metadata["page_label"] for metadata in page_metadata],
                    "original_pages": [metadata["original_page"] for metadata in page_metadata],
                },
            )
        )
    return chapter_documents


# ---------------------------------------------------------- Phase 3: saving and loading


def save_documents(documents: list[Document], path: Path) -> None:
    """Write Documents as JSON Lines and verify the file reads back identically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for doc in documents:
            record = {"page_content": doc.page_content, "metadata": doc.metadata}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    reloaded = load_documents(path)
    identical = len(reloaded) == len(documents) and all(
        saved.page_content == loaded.page_content and saved.metadata == loaded.metadata
        for saved, loaded in zip(documents, reloaded)
    )
    if not identical:
        raise ValueError(f"Round-trip check failed: {path} does not match the documents")
    logger.info("Saved %d documents to %s (%.0f KB)", len(documents), path, path.stat().st_size / 1024)


def load_documents(path: Path) -> list[Document]:
    """Read Documents saved by save_documents."""
    with path.open(encoding="utf-8") as file:
        return [
            Document(page_content=record["page_content"], metadata=record["metadata"])
            for record in map(json.loads, file)
        ]


# -------------------------------------------------------------------- Phase 4: chunking


def page_positions_of_span(page_starts: list[int], start: int, end: int) -> tuple[int, int]:
    """Positions (in a chapter's page lists) of the first and last page covering start..end."""
    first = bisect.bisect_right(page_starts, start) - 1
    last = bisect.bisect_right(page_starts, end - 1) - 1
    return first, last


def chunk_chapters(
    chapters: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Split each chapter text into chunks and label every chunk with the pages it covers.

    Chapters are split as continuous texts, so a chunk can span a page break; its
    metadata then lists both pages.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = []
    for chapter_number, chapter in enumerate(chapters, start=1):
        chapter_metadata = chapter.metadata
        for chunk in splitter.split_documents([chapter]):
            start = chunk.metadata["start_index"]
            end = start + len(chunk.page_content)
            if chapter.page_content[start:end] != chunk.page_content:
                raise ValueError(f"Chunk not found at its start_index {start} in chapter {chapter_number}")
            first, last = page_positions_of_span(chapter_metadata["page_starts"], start, end)
            chunk.metadata = {
                "source": chapter_metadata["source"],
                "chapter_number": chapter_number,
                "start_index": start,
                "page_numbers": chapter_metadata["page_numbers"][first : last + 1],
                "page_labels": chapter_metadata["page_labels"][first : last + 1],
                "original_pages": chapter_metadata["original_pages"][first : last + 1],
            }
            chunks.append(chunk)
    return chunks


def count_spanned_page_breaks(chapters: list[Document], chunks: list[Document]) -> tuple[int, int]:
    """Count page breaks inside a paragraph, and how many of them some chunk spans."""
    breaks = spanned = 0
    for chapter_number, chapter in enumerate(chapters, start=1):
        spans = [
            (chunk.metadata["start_index"], chunk.metadata["start_index"] + len(chunk.page_content))
            for chunk in chunks
            if chunk.metadata["chapter_number"] == chapter_number
        ]
        for page_start in chapter.metadata["page_starts"][1:]:
            if chapter.page_content[page_start - 1] != " ":
                continue  # joined with a blank line: no paragraph crosses this break
            breaks += 1
            spanned += any(start < page_start - 1 and end > page_start for start, end in spans)
    return breaks, spanned


# ------------------------------------------------------------------- Phase 5: embedding


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Load the sentence-transformers model on the GPU if available, else the CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Loaded %s on %s", model_name, device)
    return embedding_model


def chunks_over_token_limit(chunks: list[Document], embedding_model: HuggingFaceEmbeddings) -> list[int]:
    """Indices of chunks longer than the model reads; their ends are silently truncated."""
    sentence_model = embedding_model._client  # internal attribute, read only for the tokenizer
    token_limit = sentence_model.max_seq_length
    return [
        index
        for index, chunk in enumerate(chunks)
        if len(sentence_model.tokenizer(chunk.page_content)["input_ids"]) > token_limit
    ]


# -------------------------------------------------------------------- Phase 5: indexing


def build_or_load_index(
    chunks: list[Document], embedding_model: HuggingFaceEmbeddings, index_dir: Path
) -> tuple[FAISS, bool]:
    """Load the FAISS index saved in index_dir; build and save it only if it is missing.

    Returns the vector store and whether it was built in this call.
    """
    if (index_dir / "index.faiss").exists():
        vector_store = FAISS.load_local(
            str(index_dir),
            embedding_model,
            allow_dangerous_deserialization=True,  # index.pkl is a pickle this pipeline wrote
            distance_strategy=DISTANCE_STRATEGY,
        )
        return vector_store, False

    vector_store = FAISS.from_documents(chunks, embedding_model, distance_strategy=DISTANCE_STRATEGY)
    vector_store.save_local(str(index_dir))
    return vector_store, True


def verify_index(index_dir: Path, chunks: list[Document], embedding_model: HuggingFaceEmbeddings) -> None:
    """Reload the saved index and check it matches the chunks in memory.

    A mismatch means the index on disk was built from older chunks (it is loaded
    whenever it exists), so the caller must delete it and rebuild.
    """
    reloaded_store = FAISS.load_local(
        str(index_dir),
        embedding_model,
        allow_dangerous_deserialization=True,
        distance_strategy=DISTANCE_STRATEGY,
    )
    reloaded_index = reloaded_store.index
    vector_count = reloaded_index.ntotal
    if vector_count != len(chunks):
        raise ValueError(
            f"{index_dir} holds {vector_count} vectors for {len(chunks)} chunks; delete it and re-run"
        )

    stored_chunks = [
        reloaded_store.docstore.search(reloaded_store.index_to_docstore_id[position])
        for position in range(vector_count)
    ]
    if any(
        stored.page_content != chunk.page_content or stored.metadata != chunk.metadata
        for stored, chunk in zip(stored_chunks, chunks)
    ):
        raise ValueError(f"{index_dir} was built from different chunks; delete it and re-run")

    stored_vectors = reloaded_index.reconstruct_n(0, vector_count)
    fresh_vectors = np.array(embedding_model.embed_documents([chunk.page_content for chunk in chunks]))
    largest_difference = np.abs(stored_vectors - fresh_vectors).max()
    vector_lengths = np.linalg.norm(stored_vectors, axis=1)
    _, positions = reloaded_index.search(stored_vectors, 1)
    finds_itself = int((positions[:, 0] == np.arange(vector_count)).sum())

    if largest_difference > 1e-5 or not np.allclose(vector_lengths, 1.0, atol=1e-5):
        raise ValueError(f"Stored vectors in {index_dir} do not match the model's; delete it and re-run")
    logger.info(
        "Index verified: %d vectors of %d numbers, stored-vs-fresh difference %.1e, %d/%d find themselves",
        vector_count,
        reloaded_index.d,
        largest_difference,
        finds_itself,
        vector_count,
    )


# ------------------------------------------------------------------- Phase 6: retrieval


def make_retriever(vector_store: FAISS, k: int = TOP_K) -> VectorStoreRetriever:
    """Top-k similarity retriever: a Runnable for the Phase 7 chain."""
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def retrieve_with_scores(vector_store: FAISS, question: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """The retriever's top-k chunks with their cosine similarity, best first.

    Uses the raw inner-product score: LangChain's relevance scores report 1 - cosine
    for an inner-product index, so they rank the best chunk lowest.
    """
    return [(chunk, float(score)) for chunk, score in vector_store.similarity_search_with_score(question, k=k)]


def log_retrieval(vector_store: FAISS, question: str, k: int = TOP_K, preview_characters: int = 150) -> None:
    """Log the top-k chunks for a question with score, chapter, pages and a text preview."""
    logger.info("\nQUESTION: %s", question)
    for rank, (chunk, score) in enumerate(retrieve_with_scores(vector_store, question, k), start=1):
        preview = chunk.page_content[:preview_characters].replace("\n", " ")
        logger.info(
            "  %d. score %.3f | chapter %d | pages %s | %s ...",
            rank,
            score,
            chunk.metadata["chapter_number"],
            chunk.metadata["page_numbers"],
            preview,
        )


# ------------------------------------------------------------- Phase 7: LLM and retries


def load_api_key(variable: str = "GOOGLE_API_KEY") -> None:
    """Load the project's .env into the environment and check that the given key is set."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get(variable, "").strip()
    if not key or key == "your-key-here":
        raise RuntimeError(f"{variable} is missing. Add it to the .env file in {PROJECT_ROOT} (see .env.example).")


def make_gemini_llm(model: str = GEMINI_MODEL) -> ChatGoogleGenerativeAI:
    """Gemini chat model: deterministic, no reasoning pass, one attempt (we retry ourselves)."""
    return ChatGoogleGenerativeAI(model=model, temperature=0, thinking_budget=0, max_retries=1)


def make_groq_llm(model: str = GROQ_PRIMARY_MODEL, reasoning_effort: str = "low"):
    """A Groq chat model: deterministic, little reasoning, one attempt (we retry ourselves)."""
    from langchain_groq import ChatGroq  # optional dependency, needed only for Groq

    return ChatGroq(model=model, temperature=0, reasoning_effort=reasoning_effort, max_retries=0)


def make_groq_llm_with_fallback() -> Runnable:
    """GPT OSS 120B, falling back to Qwen 3.8 27B (separate rate limits) if a request fails."""
    return make_groq_llm(GROQ_PRIMARY_MODEL, "low").with_fallbacks([make_groq_llm(GROQ_FALLBACK_MODEL, "none")])


def http_status_of(error: BaseException | None) -> int | None:
    """The HTTP status code carried by an error or by any error it was raised from.

    Google's errors have .code, Groq's (and OpenAI-style) errors .status_code, and some
    LangChain integrations raise plain errors whose message starts with "[429]".
    """
    while error is not None:
        for attribute in ("code", "status_code"):
            value = getattr(error, attribute, None)
            if isinstance(value, int):
                return value
        match = STATUS_IN_MESSAGE.match(str(error))
        if match:
            return int(match.group(1))
        error = error.__cause__
    return None


def suggested_wait_of(error: BaseException | None) -> float | None:
    """Seconds the provider asks us to wait: Google's RetryInfo entry or a retry-after header."""
    while error is not None:
        details = getattr(error, "details", None)
        if isinstance(details, dict):
            for entry in details.get("error", {}).get("details", []):
                if entry.get("@type", "").endswith("RetryInfo"):
                    return float(entry["retryDelay"].rstrip("s"))
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None and headers.get("retry-after") is not None:
            return float(headers["retry-after"])
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
                    f"HTTP {status}: the provider asks us to wait {suggested / 3600:.1f} hours. The daily "
                    "free-tier quota for this model is used up; try again after it resets."
                ) from error
            delay = max(initial_delay_seconds * 2 ** (attempt - 1), suggested or 0) + random.uniform(0, 1)
            logger.info("HTTP %s on attempt %d; waiting %.1f s before retrying", status, attempt, delay)
            time.sleep(delay)


# ------------------------------------------------------------------- Phase 7: the chain


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


def cited_pages_of(answer: str) -> set[int]:
    """Every page cited in the answer, however the citation is spelled."""
    pages = set()
    for inside in BRACKETED.findall(answer):
        for first, last in PAGE_REFERENCE.findall(inside):
            pages.update(range(int(first), int(last or first) + 1))
    return pages


def uses_required_form(answer: str) -> bool:
    """True if the answer has citations and every one is written exactly as [p. N] or [pp. N-M]."""
    citations = [match.group(0) for match in BRACKETED.finditer(answer) if PAGE_REFERENCE.search(match.group(1))]
    return bool(citations) and all(REQUIRED_FORM.fullmatch(citation) for citation in citations)


def unsupported_citations(answer: str, source_chunks) -> tuple[set[int], set[int]]:
    """Pages cited in the answer, and those among them that no source chunk covers."""
    supplied = {page for chunk in source_chunks for page in chunk.metadata["page_numbers"]}
    cited = cited_pages_of(answer)
    return cited, cited - supplied


# ------------------------------------------------------------------ Phase 8: evaluation


def load_eval_set(path: Path = DEFAULT_EVAL_PATH) -> list[dict]:
    """The evaluation questions: id, type, question, answer, gold_pages and (factual) key_facts."""
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def score_retrieval(page_lists: list[list[int]], gold_pages, k_values=(3, 5, 10)) -> dict:
    """Where gold pages appear in a ranked list of chunk page lists: first rank, hit@k, pages found@k."""
    gold = set(gold_pages)
    gold_ranks = [rank for rank, pages in enumerate(page_lists, start=1) if gold & set(pages)]
    scores = {"first gold rank": gold_ranks[0] if gold_ranks else None}
    for k in k_values:
        found = gold & {page for pages in page_lists[:k] for page in pages}
        scores[f"hit@{k}"] = bool(found)
        scores[f"gold found@{k}"] = len(found)
    return scores


def contains_key_facts(answer: str, key_facts: list[list[str]]) -> bool:
    """True if the answer contains, for every key fact, at least one of its spellings as a whole number."""
    simplified = re.sub(r"(?<=\d),(?=\d)", "", answer.lower())  # 49,796 -> 49796
    simplified = re.sub(r"\s*:\s*", ":", simplified)  # 10.9 : 4.1 : 1 -> 10.9:4.1:1

    def stands_alone(variant: str) -> bool:
        # Not part of a longer number: no digit or point just before, no further digits after.
        return re.search(rf"(?<![\d.]){re.escape(variant)}(?!\.?\d)", simplified) is not None

    return all(any(stands_alone(variant) for variant in variants) for variants in key_facts)


# ----------------------------------------------------- whole stages: Phases 2-3 and 4-6


def prepare_corpus(pdf_path: Path = DEFAULT_PDF_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[list[Document], list[Document]]:
    """Phases 2-3: load, clean and join the PDF; save pages and chapters; return both."""
    pages = load_pages(pdf_path)
    heads = discover_running_heads(pages)
    footnote_blocks = find_footnote_blocks(pages)
    documents, dropped = clean_documents(pages, heads, footnote_blocks)
    logger.info("Kept %d of %d pages; dropped spacer pages %s", len(documents), len(pages), dropped)
    flag_continuations(documents)
    chapters = build_chapter_documents(documents)
    save_documents(documents, output_dir / PAGES_FILENAME)
    save_documents(chapters, output_dir / CHAPTERS_FILENAME)
    return documents, chapters


@dataclass
class RetrievalStack:
    """Everything Phases 4-6 produce, ready for the Phase 7 chain."""

    chapters: list[Document]
    chunks: list[Document]
    embedding_model: HuggingFaceEmbeddings
    vector_store: FAISS
    retriever: VectorStoreRetriever


def load_retriever(
    chapters_path: Path = DEFAULT_CHAPTERS_PATH, index_dir: Path = DEFAULT_INDEX_DIR, k: int = TOP_K
) -> RetrievalStack:
    """Phases 4-6: chunk the saved chapters, load (or build) and verify the index, make the retriever."""
    if not chapters_path.exists():
        raise FileNotFoundError(f"{chapters_path} not found. Run rag_utils.prepare_corpus() first.")
    chapters = load_documents(chapters_path)
    chunks = chunk_chapters(chapters)
    embedding_model = load_embedding_model()
    vector_store, _ = build_or_load_index(chunks, embedding_model, index_dir)
    verify_index(index_dir, chunks, embedding_model)
    return RetrievalStack(chapters, chunks, embedding_model, vector_store, make_retriever(vector_store, k))
