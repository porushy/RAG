"""Phase 1: fetch the India Economic Survey PDF and slice out our corpus.

Run:

    conda activate rag
    python download_data.py

This produces `data/survey_chapters.pdf`, the only input the rest of the
pipeline ever reads. Both the full download and the slice live under `data/`,
which is gitignored -- the corpus is not ours to redistribute, and a fresh
clone rebuilds it by running this script.

Why we slice instead of using the whole document
------------------------------------------------
The Survey's opening chapters are macroeconomic and mostly tables. `PyPDFLoader`
reads a PDF's text layer in reading order and has no concept of a table, so a
grid of figures arrives as a run of loose digits with the row and column labels
that gave them meaning stripped away. Embedding that produces vectors that match
nothing useful and pollute retrieval. We therefore keep the prose-heavy sectoral
and social chapters and drop the rest at the source.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from pypdf import PdfReader, PdfWriter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_URL = "https://www.indiabudget.gov.in/economicsurvey/doc/echapter.pdf"

# Page numbers as a human reads them in a PDF viewer: 1-based, both ends
# included. pypdf indexes from 0, so we convert at the point of use.
FIRST_PAGE = 276
LAST_PAGE = 468

DATA_DIR = Path(__file__).parent / "data"
FULL_PDF = DATA_DIR / "economic_survey_full.pdf"
SLICED_PDF = DATA_DIR / "survey_chapters.pdf"

# Some government servers reject requests that do not look like a browser.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def download(url: str, destination: Path) -> None:
    """Fetch `url` into `destination`, skipping the work if it is already there.

    The full Survey is a large file on a server that is not always fast, so we
    cache it. Re-running this script after a successful download costs nothing.
    """
    if destination.exists() and destination.stat().st_size > 0:
        size_mb = destination.stat().st_size / 1_048_576
        print(f"Already downloaded: {destination.name} ({size_mb:.1f} MB) -- skipping.")
        return

    print(f"Downloading {url}")
    print("  (large file; this can take a few minutes)")
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, headers=BROWSER_HEADERS)
    # `with` blocks (context managers) guarantee both the network connection and
    # the output file are closed even if the transfer raises partway through.
    # We stream the response straight to disk rather than loading it into memory.
    with urllib.request.urlopen(request, timeout=120) as response:
        with open(destination, "wb") as file_handle:
            shutil.copyfileobj(response, file_handle)

    size_mb = destination.stat().st_size / 1_048_576
    print(f"Saved {destination} ({size_mb:.1f} MB)")


def slice_pages(source: Path, destination: Path, first: int, last: int) -> None:
    """Copy pages `first`..`last` (1-based, inclusive) from `source` into a new PDF."""
    reader = PdfReader(source)
    total = len(reader.pages)
    print(f"\nFull document: {total} pages")

    if last > total:
        print(f"ERROR: requested page {last} but the document only has {total}.")
        sys.exit(1)

    writer = PdfWriter()
    # range(first - 1, last) converts 1-based inclusive page numbers into the
    # 0-based half-open indices Python uses: pages 276..468 -> indices 275..467.
    for page_index in range(first - 1, last):
        writer.add_page(reader.pages[page_index])

    with open(destination, "wb") as file_handle:
        writer.write(file_handle)

    size_mb = destination.stat().st_size / 1_048_576
    print(f"Sliced pages {first}-{last} ({last - first + 1} pages) "
          f"-> {destination} ({size_mb:.1f} MB)")


def verify_text_layer(pdf_path: Path) -> None:
    """Confirm the slice contains real, selectable text rather than page images.

    This is the acceptance check that matters. A PDF made by scanning paper holds
    pictures of words: every extraction returns an empty string, and no amount of
    cleaning or chunking downstream can recover anything. Better to discover that
    now than after building an index over nothing.
    """
    reader = PdfReader(pdf_path)
    page_count = len(reader.pages)
    lengths = [len(page.extract_text() or "") for page in reader.pages]

    empty = sum(1 for length in lengths if length == 0)
    average = sum(lengths) / page_count if page_count else 0

    print(f"\nText-layer check over {page_count} pages")
    print(f"  pages with zero extractable text : {empty}")
    print(f"  mean characters per page         : {average:,.0f}")
    print(f"  min / max characters per page    : {min(lengths):,} / {max(lengths):,}")

    if empty > page_count * 0.1:
        print("\n  WARNING: many pages extracted nothing. This may be a scanned")
        print("  document, in which case we need a different source.")
    else:
        print("\n  OK: the document has a real text layer.")

    # Show the top of the first and last page so the chapter boundaries are
    # visible and the page range can be confirmed by eye.
    for label, page in (("FIRST", reader.pages[0]), ("LAST", reader.pages[-1])):
        text = (page.extract_text() or "").strip()
        preview = " ".join(text.split())[:300]
        print(f"\n--- {label} PAGE OF SLICE ---\n{preview}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download(SOURCE_URL, FULL_PDF)
    slice_pages(FULL_PDF, SLICED_PDF, FIRST_PAGE, LAST_PAGE)
    verify_text_layer(SLICED_PDF)


if __name__ == "__main__":
    main()
