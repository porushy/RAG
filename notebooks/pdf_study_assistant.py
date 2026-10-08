"""Load, clean and save the Economic Survey chapters for the PDF study assistant.

Pipeline (Phases 1-3 of pdf_study_assistant.ipynb, which documents the evidence
behind every rule below):

    1. Load the PDF one Document per page.
    2. Discover running heads and locate each page's footnote block.
    3. Clean each page (footnotes + eight ordered steps) and drop blank spacer pages.
    4. Flag pages that continue a paragraph broken off by the previous page.
    5. Join each chapter into one continuous text, recording where every page starts,
       so a chunk can span a page break and still be cited by its page range.
    6. Save pages and chapters as JSON Lines and verify the round trips.

Usage (from the project root or from notebooks/):

    python notebooks/pdf_study_assistant.py [--pdf PATH] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import bisect
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PAGES_FILENAME = "cleaned_documents.jsonl"
CHAPTERS_FILENAME = "cleaned_chapters.jsonl"

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
APPARATUS_PATTERN = re.compile(r"^\s*(?:Sources?|Notes?)\s*:", re.IGNORECASE)
HYPHEN_LINEBREAK = re.compile(r"([A-Za-z])-[ \t]*\n[ \t]*([A-Za-z])")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
FOOTNOTE_AFTER_WORD = re.compile(r"(?<=[a-z])\.\d{1,2}(?=\s+[A-Z])")
# Four digits, so that paragraph markers such as 8.44 and 10.98 cannot match.
FOOTNOTE_AFTER_YEAR = re.compile(r"(?<=\d{4})\.\d{1,2}(?=\s+[A-Z])")
# The document's own paragraph numbers: chapters 6-10, followed by text.
PARAGRAPH_MARKER = re.compile(r"(?m)^[ \t]*((?:6|7|8|9|10)\.\d{1,3})\.?[ \t]+(?=[A-Za-z(])")
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


# ----------------------------------------------------------------- corpus-wide passes


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


def find_footnote_blocks(pages: list[Document]) -> dict[int, tuple[int, int, int]]:
    """Locate each page's footnote block by following the document's footnote numbering.

    A block must start with exactly the next expected footnote number (numbering
    restarts at each chapter); body lines or chart values that merely start with a
    number are therefore not mistaken for footnotes.

    Returns {page_index: (first_line_index, first_number, last_number)}.
    """
    blocks = {}
    expected_number = 1
    for page_index, page in enumerate(pages):
        lines = page.page_content.splitlines()
        if CHAPTER_START.search("\n".join(lines[:4])):
            expected_number = 1
        candidates = [index for index, line in enumerate(lines) if footnote_number(line) == expected_number]
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


# -------------------------------------------------------------------------- cleaning


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


def clean_page(text: str, heads: set[str], footnote_block: tuple[int, int, int] | None = None) -> str:
    """Run Step 0 and the eight cleaning steps. The order matters."""
    text = remove_footnote_block(text, footnote_block)  # 0  needs the original line layout
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


# ------------------------------------------------------------ joining across pages


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


def pages_of_span(chapter_metadata: dict, start: int, end: int) -> list[int]:
    """Page numbers covered by characters start..end of a chapter text."""
    page_starts = chapter_metadata["page_starts"]
    first = bisect.bisect_right(page_starts, start) - 1
    last = bisect.bisect_right(page_starts, end - 1) - 1
    return chapter_metadata["page_numbers"][first : last + 1]


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
        raise ValueError(f"Round-trip check failed: {path} does not match the documents")
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
