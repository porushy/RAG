"""Load, clean and save the Economic Survey chapters for the PDF study assistant.

Pipeline (Phases 1-3 of pdf_study_assistant.ipynb, which documents the evidence
behind every rule below):

    1. Load the PDF one Document per page.
    2. Discover running heads from the data.
    3. Clean each page in eight ordered steps and drop blank spacer pages.
    4. Save the cleaned pages as JSON Lines and verify the round trip.

Usage (from the project root or from notebooks/):

    python notebooks/pdf_study_assistant.py [--pdf PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "survey_chapters.pdf"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_documents.jsonl"

FIRST_PAGE_IN_ORIGINAL_PDF = 276  # first page of the slice in the full Survey (download_data.py)
BOILERPLATE_THRESHOLD = 10  # a first line repeated on this many pages is a running head
HEAD_LINES_TO_CHECK = 3  # running heads are only removed near the top of a page
MINIMUM_LETTER_RATIO = 0.20  # lines below this are flattened chart data

# The notice printed on the spacer pages between chapters; only these pages are dropped.
BLANK_PAGE_PATTERN = re.compile(r"\s*This page has been left blank\.?\s*", re.IGNORECASE)
APPARATUS_PATTERN = re.compile(r"^\s*(?:Sources?|Notes?)\s*:", re.IGNORECASE)
HYPHEN_LINEBREAK = re.compile(r"([A-Za-z])-[ \t]*\n[ \t]*([A-Za-z])")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
FOOTNOTE_AFTER_WORD = re.compile(r"(?<=[a-z])\.\d{1,2}(?=\s+[A-Z])")
# Four digits, so that paragraph markers such as 8.44 and 10.98 cannot match.
FOOTNOTE_AFTER_YEAR = re.compile(r"(?<=\d{4})\.\d{1,2}(?=\s+[A-Z])")
# The document's own paragraph numbers: chapters 6-10, followed by text.
PARAGRAPH_MARKER = re.compile(r"(?m)^[ \t]*((?:6|7|8|9|10)\.\d{1,3})\.?[ \t]+(?=[A-Za-z(])")


# --------------------------------------------------------------------------- loading


def load_pages(pdf_path: Path) -> list[Document]:
    """Read the PDF into one Document per page."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"{pdf_path} not found. Run `python download_data.py` from the project root first."
        )
    pages = PyPDFLoader(str(pdf_path)).load()
    logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)
    return pages


# -------------------------------------------------------------------------- cleaning


def letter_ratio(line: str) -> float:
    """Fraction of a line's characters that are alphabetic letters."""
    if not line:
        return 0.0
    return sum(character.isalpha() for character in line) / len(line)


def discover_running_heads(pages: list[Document], threshold: int = BOILERPLATE_THRESHOLD) -> set[str]:
    """Return first lines that repeat across many pages (headers, not prose)."""
    first_lines = []
    for page in pages:
        non_empty = [line.strip() for line in page.page_content.splitlines() if line.strip()]
        if non_empty:
            first_lines.append(non_empty[0])
    counts = Counter(first_lines)
    return {line for line, count in counts.items() if count >= threshold}


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


def clean_page(text: str, heads: set[str]) -> str:
    """Run the eight cleaning steps. The order matters."""
    text = remove_running_heads(text, heads)  # 1
    text = remove_junk_lines(text)  # 2  must precede 7: removes chart values shaped like markers
    text = remove_apparatus(text)  # 3
    text = join_hyphenated_linebreaks(text)  # 4  must precede 5: reunites split URLs
    text = strip_urls(text)  # 5
    text = repair_footnote_markers(text)  # 6
    text = mark_paragraphs(text)  # 7
    text = collapse_whitespace(text)  # 8
    return text


def is_blank_spacer(text: str) -> bool:
    """True only when the whole page is the 'This page has been left blank' notice."""
    return BLANK_PAGE_PATTERN.fullmatch(text) is not None


def clean_documents(pages: list[Document], heads: set[str]) -> tuple[list[Document], list[int]]:
    """Clean every page; return the kept Documents and the indices of dropped spacer pages."""
    cleaned, dropped = [], []
    for page_index, page in enumerate(pages):
        text = clean_page(page.page_content, heads)
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


# ----------------------------------------------------------------------- persistence


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
        raise ValueError(f"Round-trip check failed: {path} does not match the cleaned documents")
    logger.info("Saved %d documents to %s (%.0f KB)", len(documents), path, path.stat().st_size / 1024)


def load_documents(path: Path) -> list[Document]:
    """Read Documents saved by save_documents."""
    with path.open(encoding="utf-8") as file:
        return [
            Document(page_content=record["page_content"], metadata=record["metadata"])
            for record in map(json.loads, file)
        ]


# ------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH, help="source PDF")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="cleaned JSONL file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pages = load_pages(args.pdf)
    heads = discover_running_heads(pages)
    logger.info("Discovered %d running heads", len(heads))

    documents, dropped = clean_documents(pages, heads)
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

    save_documents(documents, args.output)


if __name__ == "__main__":
    main()
