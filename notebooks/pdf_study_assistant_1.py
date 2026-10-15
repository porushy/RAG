"""Load, clean and save the Economic Survey chapters for the PDF study assistant.

Pipeline (Phases 1-3 of pdf_study_assistant.ipynb, which documents the evidence behind
every rule; the functions themselves live in rag_utils.py):

    1. Load the PDF one Document per page.
    2. Discover running heads and locate each page's footnote block.
    3. Clean each page (footnotes + nine ordered steps, ending with one sentence per
       line) and drop blank spacer pages.
    4. Flag pages that continue a paragraph broken off by the previous page.
    5. Join each chapter into one continuous text, recording where every page starts,
       so a chunk can span a page break and still be cited by its page range.
    6. Save pages and chapters as JSON Lines and verify the round trips.

Phase 4 onwards (chunking, embedding, indexing) is in pdf_study_assistant_2.py, which
reads the chapters file written here.

Usage (from the project root or from notebooks/):

    python notebooks/pdf_study_assistant_1.py [--pdf PATH] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # rag_utils.py is in the project root

# Re-exported as well: the Part 3 notebooks import these names from this module.
from rag_utils import (  # noqa: E402
    CHAPTERS_FILENAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDF_PATH,
    PAGES_FILENAME,
    PROJECT_ROOT,
    build_chapter_documents,
    clean_documents,
    discover_running_heads,
    find_footnote_blocks,
    flag_continuations,
    load_documents,
    load_pages,
    save_documents,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH, help="source PDF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="folder for the JSONL files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pages = load_pages(args.pdf)
    heads = discover_running_heads(pages)
    footnote_blocks = find_footnote_blocks(pages)
    footnote_count = sum(last - first + 1 for _, first, last in footnote_blocks.values())
    logger.info("Discovered %d running heads; %d footnotes on %d pages", len(heads), footnote_count, len(footnote_blocks))

    documents, dropped = clean_documents(pages, heads, footnote_blocks)
    characters_before = sum(len(page.page_content) for page in pages)
    characters_after = sum(len(doc.page_content) for doc in documents)
    paragraph_breaks = "\n\n".join(doc.page_content for doc in documents).count("\n\n")
    logger.info("Kept %d of %d pages; dropped %s", len(documents), len(pages), dropped)
    logger.info(
        "Characters %s -> %s (%.1f%% removed); %d paragraph breaks",
        f"{characters_before:,}",
        f"{characters_after:,}",
        100 * (characters_before - characters_after) / characters_before,
        paragraph_breaks,
    )

    continuations = flag_continuations(documents)
    chapters = build_chapter_documents(documents)
    logger.info("%d pages continue the previous page; joined into %d chapters", continuations, len(chapters))

    save_documents(documents, args.output_dir / PAGES_FILENAME)
    save_documents(chapters, args.output_dir / CHAPTERS_FILENAME)


if __name__ == "__main__":
    main()
