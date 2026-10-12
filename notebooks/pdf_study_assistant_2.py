"""Chunk, embed and index the cleaned Economic Survey chapters for the PDF study assistant.

Pipeline (Phases 4-5 of pdf_study_assistant_part2.ipynb, which explains and checks
every step below):

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
import bisect
import logging
import time
from pathlib import Path

import numpy as np
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_study_assistant_1 import CHAPTERS_FILENAME, DEFAULT_OUTPUT_DIR, PROJECT_ROOT, load_documents

logger = logging.getLogger(__name__)

DEFAULT_CHAPTERS_PATH = DEFAULT_OUTPUT_DIR / CHAPTERS_FILENAME
DEFAULT_INDEX_DIR = PROJECT_ROOT / "faiss_index"

CHUNK_SIZE = 1000  # maximum characters per chunk
CHUNK_OVERLAP = 200  # at most this many characters repeated from the previous chunk
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Unit vectors: inner product = cosine similarity. LangChain's default is Euclidean, and
# the saved index does not record the strategy, so it is passed on build and on load.
DISTANCE_STRATEGY = DistanceStrategy.MAX_INNER_PRODUCT
TOP_K = 5  # chunks returned per question


# -------------------------------------------------------------------------- chunking


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


# ------------------------------------------------------------------------- embedding


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


# -------------------------------------------------------------------------- indexing


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


# ------------------------------------------------------------------------- retrieval


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


# ------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chapters", type=Path, default=DEFAULT_CHAPTERS_PATH, help="chapters JSONL file")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="folder for the FAISS index")
    parser.add_argument("--question", action="append", default=[], help="question to retrieve chunks for (repeatable)")
    parser.add_argument("--k", type=int, default=TOP_K, help="chunks to retrieve per question")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")  # quiet HTTP and FAISS loader logs
    logger.setLevel(logging.INFO)

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
