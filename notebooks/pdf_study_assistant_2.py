"""Chunk, embed and index the cleaned Economic Survey chapters for the PDF study assistant.

Pipeline (Phases 4-6 of pdf_study_assistant_part2.ipynb, which explains and checks every
step; the functions themselves live in rag_utils.py):

    1. Read the chapter texts written by pdf_study_assistant_1.py.
    2. Split each chapter into chunks (1,000 characters, up to 200 overlap) and give
       every chunk the page range its text came from.
    3. Embed the chunks with all-MiniLM-L6-v2 (GPU if available, else CPU), normalised
       so that inner product equals cosine similarity.
    4. Load the FAISS index from disk, or build and save it if it is missing.
    5. Verify the index against the chunks in memory.
    6. Optionally answer --question with the top-k chunks, their cosine scores and pages
       (retrieval only, no LLM; Phase 6).

Usage (from the project root or from notebooks/):

    python notebooks/pdf_study_assistant_2.py [--chapters PATH] [--index-dir DIR] [--question TEXT ...] [--k K]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # rag_utils.py is in the project root

# Re-exported as well: the Part 3 notebooks import these names from this module.
from rag_utils import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_CHAPTERS_PATH,
    DEFAULT_INDEX_DIR,
    DISTANCE_STRATEGY,
    EMBEDDING_MODEL_NAME,
    TOP_K,
    build_or_load_index,
    chunk_chapters,
    chunks_over_token_limit,
    count_spanned_page_breaks,
    load_documents,
    load_embedding_model,
    log_retrieval,
    make_retriever,
    retrieve_with_scores,
    verify_index,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chapters", type=Path, default=DEFAULT_CHAPTERS_PATH, help="chapters JSONL file")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="folder for the FAISS index")
    parser.add_argument("--question", action="append", default=[], help="question to retrieve chunks for (repeatable)")
    parser.add_argument("--k", type=int, default=TOP_K, help="chunks to retrieve per question")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")  # quiet HTTP and FAISS loader logs
    for name in (__name__, "rag_utils"):
        logging.getLogger(name).setLevel(logging.INFO)

    if not args.chapters.exists():
        raise FileNotFoundError(f"{args.chapters} not found. Run `python notebooks/pdf_study_assistant_1.py` first.")
    chapters = load_documents(args.chapters)

    chunks = chunk_chapters(chapters)
    two_page_chunks = sum(len(chunk.metadata["page_numbers"]) > 1 for chunk in chunks)
    breaks, spanned = count_spanned_page_breaks(chapters, chunks)
    logger.info("Split %d chapters into %d chunks (%d span two pages)", len(chapters), len(chunks), two_page_chunks)
    logger.info("Paragraphs crossing a page break: %d; spanned by a chunk: %d", breaks, spanned)

    embedding_model = load_embedding_model()
    truncated = chunks_over_token_limit(chunks, embedding_model)
    logger.info("Chunks over the model's token limit (end truncated): %d %s", len(truncated), truncated)

    start_time = time.perf_counter()
    vector_store, built = build_or_load_index(chunks, embedding_model, args.index_dir)
    logger.info(
        "Index %s in %.1f s: %s",
        "built and saved" if built else "loaded from disk",
        time.perf_counter() - start_time,
        args.index_dir,
    )
    verify_index(args.index_dir, chunks, embedding_model)

    for question in args.question:
        log_retrieval(vector_store, question, args.k)


if __name__ == "__main__":
    main()
