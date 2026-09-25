from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader

# Work out the project root. Jupyter may be launched from either the project
# root or from inside notebooks/, so we handle both.
project_root = Path.cwd()
if project_root.name == "notebooks":
    project_root = project_root.parent

pdf_path = project_root / "data" / "survey_chapters.pdf"

print(f"Project root : {project_root}")
print(f"PDF path     : {pdf_path}")

# Fail early and clearly if the corpus is missing, rather than letting an
# obscure error appear several cells later.
if not pdf_path.exists():
    raise FileNotFoundError(
        f"{pdf_path} not found. Run `python download_data.py` from the project "
        "root first -- it downloads the Survey and slices out our five chapters."
    )

size_mb = pdf_path.stat().st_size / 1_048_576
print(f"PDF size     : {size_mb:.1f} MB")

# Create the loader and read the file. This takes a few seconds for 193 pages.
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

print(f"Number of Document objects returned: {len(pages)}")
print(f"Type of each item                  : {type(pages[0]).__name__}")

# Show the structure of a single Document so the two parts are concrete.
first_page = pages[0]
print(f"\nAttributes of one Document:")
print(f"  .page_content -> {type(first_page.page_content).__name__}, "
      f"{len(first_page.page_content)} characters")
print(f"  .metadata     -> {first_page.metadata}")

print(pages[0].page_content)

print(pages[0].metadata)

# For each page, count how many characters of text were extracted.
character_counts = [len(page.page_content) for page in pages]

length_summary = pd.Series(character_counts, name="characters_per_page")
print(length_summary.describe().round(1).to_string())

# Count pages that are suspiciously short -- likely blanks or chapter dividers.
nearly_empty = [
    index for index, count in enumerate(character_counts) if count < 100
]
print(f"\nPages with fewer than 100 characters: {len(nearly_empty)}")
print(f"  (0-based indices: {nearly_empty})")
print(f"  (human page numbers: {[index + 1 for index in nearly_empty]})")

total_characters = sum(character_counts)
print(f"\nTotal characters in corpus: {total_characters:,}")
print(f"Roughly {total_characters / 5:,.0f} words, or about "
      f"{total_characters / 4:,.0f} tokens.")

figure, axes = plt.subplots(figsize=(10, 4))

axes.hist(character_counts, bins=40, color="#4C78A8", edgecolor="white")
axes.axvline(
    length_summary.mean(),
    color="#E45756",
    linestyle="--",
    label=f"mean = {length_summary.mean():,.0f}",
)

axes.set_xlabel("Characters extracted from the page")
axes.set_ylabel("Number of pages")
axes.set_title("Distribution of page lengths across the 193-page corpus")
axes.legend()

plt.tight_layout()
plt.show()

# Four pages sampled from different parts of the corpus:
# a chapter opening, two mid-chapter prose pages, and a blank spacer.
pages_to_show = [0, 20, 33, 160]

for page_index in pages_to_show:
    page = pages[page_index]
    human_page_number = page_index + 1
    raw_text = page.page_content

    print("=" * 78)
    print(f"PAGE INDEX {page_index}  (human page {human_page_number} of the slice)")
    print(f"metadata: {page.metadata}")
    print(f"length  : {len(raw_text)} characters")
    print("=" * 78)

    # repr() makes invisible characters visible: newlines show as \n.
    print("\n--- RAW, WITH ESCAPE CHARACTERS SHOWN ---")
    print(repr(raw_text))

    # And now the same text rendered normally, so it reads as prose.
    print("\n--- SAME TEXT, RENDERED ---")
    print(raw_text)
    print("\n")

from collections import Counter

first_lines = []
last_lines = []

for page in pages:
    # Keep only lines that contain something other than whitespace.
    lines = [line.strip() for line in page.page_content.splitlines() if line.strip()]
    if not lines:
        continue  # blank page -- nothing to record
    first_lines.append(lines[0])
    last_lines.append(lines[-1])

print(f"Pages contributing lines: {len(first_lines)} of {len(pages)}\n")

print("TEN MOST COMMON FIRST LINES")
print("-" * 70)
for line_text, occurrences in Counter(first_lines).most_common(10):
    print(f"{occurrences:>4} x  {line_text[:64]!r}")

print("\nTEN MOST COMMON LAST LINES")
print("-" * 70)
for line_text, occurrences in Counter(last_lines).most_common(10):
    print(f"{occurrences:>4} x  {line_text[:64]!r}")

# Join every page into one long string so we can count across the whole corpus.
corpus_text = "\n".join(page.page_content for page in pages)

double_newlines = corpus_text.count("\n\n")
single_newlines = corpus_text.count("\n")
pages_with_blank_line = sum(1 for page in pages if "\n\n" in page.page_content)

print(f"Blank lines (two newlines in a row) : {double_newlines:,}")
print(f"Single newlines                     : {single_newlines:,}")
print(f"Pages with at least one blank line  : {pages_with_blank_line} / {len(pages)}")

digit_fractions = []
for page_index, page in enumerate(pages):
    text = page.page_content
    if not text.strip():
        continue  # blank pages have no meaningful ratio
    digit_count = sum(character.isdigit() for character in text)
    digit_fractions.append((digit_count / len(text), page_index))

digit_fractions.sort(reverse=True)

print("The eight most digit-heavy pages in the corpus:")
for fraction, page_index in digit_fractions[:8]:
    print(f"   page index {page_index:>3}   {fraction:6.1%} digits   "
          f"{len(pages[page_index].page_content):>5} characters")

over_40 = [index for fraction, index in digit_fractions if fraction > 0.40]
over_25 = [index for fraction, index in digit_fractions if fraction > 0.25]

print(f"\nPages above the brief's 40% threshold : {len(over_40)}")
print(f"Pages above even a lenient 25%        : {len(over_25)}")
print(f"Highest digit fraction anywhere       : {digit_fractions[0][0]:.1%}")

def letter_ratio(line: str) -> float:
    """Fraction of a line's characters that are alphabetic letters."""
    if not line:
        return 0.0
    return sum(character.isalpha() for character in line) / len(line)


# Page index 88 was the most digit-dense; look at what is actually on it.
print("A REAL CHART PAGE, AS EXTRACTED (first 900 characters)")
print("=" * 78)
print(pages[88].page_content[:900])

# Now measure every line in the corpus.
all_lines = [
    line.strip()
    for page in pages
    for line in page.page_content.splitlines()
    if line.strip()
]

junk_candidates = [line for line in all_lines if letter_ratio(line) < 0.20]
with_letters = [line for line in junk_candidates if any(c.isalpha() for c in line)]

print("\n" + "=" * 78)
print(f"Total non-empty lines in corpus            : {len(all_lines):,}")
print(f"Lines below 20% letters (deletion targets) : {len(junk_candidates):,} "
      f"({len(junk_candidates) / len(all_lines):.1%})")
print(f"  ...of which contain ANY letter at all    : {len(with_letters)}")

print("\nThe longest deletion targets -- if we were about to destroy a real")
print("sentence, it would show up among the longest lines:")
for line in sorted(junk_candidates, key=len, reverse=True)[:10]:
    print(f"   {len(line):>3} chars, {letter_ratio(line):>4.0%} letters   {line[:70]!r}")

import re
from collections import Counter

# How often does each alphabetic word appear as a standalone word?
vocabulary = Counter(
    word.lower() for word in re.findall(r"\b[a-zA-Z]{2,}\b", corpus_text)
)

# Find every occurrence of a hyphen at a line end continuing on the next line.
hyphen_pairs = re.findall(r"([A-Za-z]+)-\s*\n\s*([A-Za-z]+)", corpus_text)
print(f"Occurrences of 'word-<newline>word': {len(hyphen_pairs)}\n")

real_compounds, soft_splits, unclear = [], [], []
for left_part, right_part in hyphen_pairs:
    standalone_uses = vocabulary[right_part.lower()]
    if standalone_uses >= 3:
        real_compounds.append((left_part, right_part, standalone_uses))
    elif standalone_uses == 0:
        soft_splits.append((left_part, right_part, standalone_uses))
    else:
        unclear.append((left_part, right_part, standalone_uses))

print(f"REAL COMPOUND (right part is a common standalone word) : {len(real_compounds)}")
for left, right, count in real_compounds[:6]:
    print(f"     {left}-{right:<14} ('{right}' appears {count}x on its own)")

print(f"\nSOFT SPLIT (right part never appears alone)            : {len(soft_splits)}")
for left, right, count in soft_splits[:6]:
    print(f"     {left}-{right}  would wrongly become '{left}{right}'")

print(f"\nUNCLEAR (right part appears once or twice)             : {len(unclear)}")
for left, right, count in unclear[:6]:
    print(f"     {left}-{right:<14} ('{right}' appears {count}x)")

lines_only = [line.strip() for line in corpus_text.splitlines() if line.strip()]

# A marker WITH text after it on the same line -- a genuine paragraph opening.
genuine_marker = re.compile(r"^(6|7|8|9|10)\.\d{1,3}\.?\s+[A-Za-z(]")
genuine = [line for line in lines_only if genuine_marker.match(line)]

# A line that is ONLY a number of the form N.M and nothing else.
bare_number = re.compile(r"^\d{1,2}\.\d{1,3}\.?$")
bare = [line for line in lines_only if bare_number.match(line)]

print(f"Genuine paragraph markers (number + text on same line) : {len(genuine)}")
for line in genuine[:4]:
    print(f"    {letter_ratio(line):>4.0%} letters   {line[:76]!r}")

print(f"\nLines that are a bare 'N.M' number and nothing else    : {len(bare)}")
print(f"    {bare[:16]}")
print("    ^ these are CHART AXIS VALUES, not paragraph numbers --")
print("      but 6.6, 6.2 and 6.3 are indistinguishable by shape alone.")

# Does the junk filter endanger the genuine markers?
genuine_ratios = [letter_ratio(line) for line in genuine]
print(f"\nLowest letter ratio among genuine markers : {min(genuine_ratios):.0%}")
print(f"Junk-deletion threshold                   : 20%")

# THE ORDERING TEST: apply junk removal first, then look for markers.
survivors = [line for line in lines_only if letter_ratio(line) >= 0.20]
fakes_left = [line for line in survivors if bare_number.match(line)]
real_left = [line for line in survivors if genuine_marker.match(line)]

print("\n" + "=" * 70)
print("ORDERING TEST: remove junk lines FIRST, then look for markers")
print("=" * 70)
print(f"  chart-value impostors surviving : {len(fakes_left)}  (was {len(bare)})")
print(f"  genuine markers surviving       : {len(real_left)}  (was {len(genuine)})")

import unicodedata

print("FOOTNOTE AND CITATION CONTAMINATION")
print("-" * 70)
glued_markers = re.findall(r"[a-z]{3,}\.\d{1,2}\s+[A-Z]", corpus_text)
embedded_urls = re.findall(r"https?://\S+", corpus_text)
apparatus_lines = re.findall(r"(?:^|\n)\s*(?:Source|Note)s?\s*:", corpus_text)

print(f"  superscript markers glued to sentence ends : {len(glued_markers)}")
print(f"     examples: {glued_markers[:6]}")
print(f"  URLs embedded in the text                  : {len(embedded_urls)}")
print(f"     example : {embedded_urls[0][:76] if embedded_urls else '-'}")
print(f"  'Source:' / 'Note:' apparatus lines        : {len(apparatus_lines)}")

print("\nNON-ASCII CHARACTERS PRESENT")
print("-" * 70)
non_ascii = Counter(character for character in corpus_text if ord(character) > 127)
for character, occurrences in non_ascii.most_common(10):
    try:
        character_name = unicodedata.name(character)
    except ValueError:
        character_name = "UNNAMED (private use area)"
    print(f"  U+{ord(character):04X}  {occurrences:>5}x  {character_name}")

broken_glyphs = len(re.findall(r"\(cid:\d+\)", corpus_text))
print(f"\n  broken-font '(cid:NN)' artefacts: {broken_glyphs}")

# Collect the first non-empty line of every page.
page_first_lines = []
for page in pages:
    non_empty = [line.strip() for line in page.page_content.splitlines() if line.strip()]
    if non_empty:
        page_first_lines.append(non_empty[0])

first_line_counts = Counter(page_first_lines)

# A line appearing on 10+ pages is boilerplate, not prose.
BOILERPLATE_THRESHOLD = 10
running_heads = {
    line for line, count in first_line_counts.items()
    if count >= BOILERPLATE_THRESHOLD
}

print(f"Running heads discovered automatically ({len(running_heads)} distinct):\n")
for head in sorted(running_heads, key=lambda h: -first_line_counts[h]):
    print(f"  {first_line_counts[head]:>3} pages   {head!r}")

print(f"\nTotal pages covered: {sum(first_line_counts[h] for h in running_heads)} of {len(pages)}")

import re

# Lines that are chart apparatus rather than content.
APPARATUS_PATTERN = re.compile(r"^\s*(?:Sources?|Notes?)\s*:", re.IGNORECASE)


def remove_running_heads(text, heads, lines_to_check=3):
    """Delete boilerplate header lines, but only near the top of the page.

    The restriction matters: 'Environment and Climate Change' is a running head AND
    an ordinary phrase that occurs in body text. Position disambiguates them.
    """
    kept_lines = []
    for position, line in enumerate(text.splitlines()):
        if position < lines_to_check and line.strip() in heads:
            continue  # boilerplate near the top -- drop it
        kept_lines.append(line)
    return "\n".join(kept_lines)


def remove_junk_lines(text, minimum_letter_ratio=0.20):
    """Delete lines that are almost entirely numbers -- flattened chart data."""
    kept_lines = [
        line for line in text.splitlines()
        if not line.strip() or letter_ratio(line.strip()) >= minimum_letter_ratio
    ]
    return "\n".join(kept_lines)


def remove_apparatus(text):
    """Delete 'Source:' and 'Note:' lines that belong to charts, not to the argument."""
    return "\n".join(
        line for line in text.splitlines() if not APPARATUS_PATTERN.match(line)
    )


# Demonstrate all three on the chart page from Phase 2.
demo = pages[88].page_content
after_structural = remove_apparatus(remove_junk_lines(
    remove_running_heads(demo, running_heads)))

print("BEFORE (first 460 characters)")
print("-" * 78)
print(demo[:460])
print("\nAFTER STEPS 1-3 (first 460 characters)")
print("-" * 78)
print(after_structural[:460])

HYPHEN_LINEBREAK = re.compile(r"([A-Za-z])-[ \t]*\n[ \t]*([A-Za-z])")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Footnote marker fused to the end of a word: "livelihoods.2 A"
FOOTNOTE_AFTER_WORD = re.compile(r"(?<=[a-z])\.\d{1,2}(?=\s+[A-Z])")
# Footnote marker fused to a year: "2024.37 From".  FOUR digits required so that
# paragraph markers such as "8.44" and "10.98" cannot possibly match.
FOOTNOTE_AFTER_YEAR = re.compile(r"(?<=\d{4})\.\d{1,2}(?=\s+[A-Z])")


def join_hyphenated_linebreaks(text):
    """Rejoin a compound split across lines, KEEPING its hyphen.

    Phase 2 Test 4 found 124 such breaks and zero genuine word-splits, so removing
    the hyphen would corrupt real words ('MSP-based' -> 'MSPbased').
    """
    return HYPHEN_LINEBREAK.sub(r"\1-\2", text)


def strip_urls(text):
    """Remove web addresses, which carry no meaning an embedding can use."""
    return URL_PATTERN.sub(" ", text)


def repair_footnote_markers(text):
    """Detach superscript footnote numbers that fused to the preceding word."""
    text = FOOTNOTE_AFTER_WORD.sub(".", text)
    return FOOTNOTE_AFTER_YEAR.sub(".", text)


# Prove the year rule cannot damage a paragraph marker.
print("SAFETY CHECK -- the year pattern must not touch paragraph markers:")
for sample in ["8.44. Bengaluru, Delhi", "10.98 The chapter", "6.1 Agriculture",
               "in global filings in 2024.37 From FY20", "grew 1.5 times"]:
    print(f"   {sample[:38]:<40} -> {repair_footnote_markers(sample)[:38]!r}")

# Only chapters 6-10, and only when real text follows on the same line.
PARAGRAPH_MARKER = re.compile(
    r"(?m)^[ \t]*((?:6|7|8|9|10)\.\d{1,3})\.?[ \t]+(?=[A-Za-z(])"
)


def mark_paragraphs(text):
    """Insert a blank line before each of the document's own paragraph numbers."""
    return PARAGRAPH_MARKER.sub(r"\n\n\1 ", text)


def collapse_whitespace(text):
    """Turn line-wrap newlines into spaces while preserving paragraph breaks."""
    tidied_paragraphs = []
    for paragraph in text.split("\n\n"):
        # Inside a paragraph, every newline is a printing artefact, not meaning.
        single_line = paragraph.replace("\n", " ")
        single_line = re.sub(r"[ \t]+", " ", single_line).strip()
        if single_line:
            tidied_paragraphs.append(single_line)
    return "\n\n".join(tidied_paragraphs)


# Demonstrate on the structurally-cleaned chart page from earlier.
demo_repaired = repair_footnote_markers(
    strip_urls(join_hyphenated_linebreaks(after_structural))
)
demo_final = collapse_whitespace(mark_paragraphs(demo_repaired))

print("AFTER ALL EIGHT STEPS")
print("-" * 78)
print(demo_final[:700])
print(f"\n\nParagraph breaks on this page: {demo_final.count(chr(10) + chr(10))}")

from langchain_core.documents import Document

MINIMUM_PAGE_CHARACTERS = 200
FIRST_PAGE_IN_ORIGINAL_PDF = 276  # from download_data.py


def clean_page(text, heads):
    """Run the eight cleaning steps. THE ORDER IS LOAD-BEARING -- see Phase 2 Test 5."""
    text = remove_running_heads(text, heads)      # 1  boilerplate headers
    text = remove_junk_lines(text)                # 2  chart debris + 151 impostors
    text = remove_apparatus(text)                 # 3  Source:/Note: lines
    text = join_hyphenated_linebreaks(text)       # 4  before URLs: reunites split links
    text = strip_urls(text)                       # 5
    text = repair_footnote_markers(text)          # 6
    text = mark_paragraphs(text)                  # 7  MUST follow step 2
    text = collapse_whitespace(text)              # 8
    return text


cleaned_documents = []
dropped_pages = []

for page_index, page in enumerate(pages):
    cleaned_text = clean_page(page.page_content, running_heads)

    if len(cleaned_text) < MINIMUM_PAGE_CHARACTERS:
        dropped_pages.append((page_index, len(cleaned_text)))
        continue

    metadata = dict(page.metadata)  # copy so we never mutate the original
    metadata["page_number"] = page_index + 1
    metadata["original_page"] = page_index + FIRST_PAGE_IN_ORIGINAL_PDF
    cleaned_documents.append(Document(page_content=cleaned_text, metadata=metadata))

characters_before = sum(len(page.page_content) for page in pages)
characters_after = sum(len(doc.page_content) for doc in cleaned_documents)

print("PHASE 3 RESULTS")
print("=" * 62)
print(f"Pages in      : {len(pages)}")
print(f"Pages out     : {len(cleaned_documents)}")
print(f"Pages dropped : {len(dropped_pages)}  -> indices {[i for i, _ in dropped_pages]}")
print()
print(f"Characters before : {characters_before:,}")
print(f"Characters after  : {characters_after:,}")
print(f"Removed           : {characters_before - characters_after:,} "
      f"({(characters_before - characters_after) / characters_before:.1%})")

cleaned_corpus = "\n\n".join(doc.page_content for doc in cleaned_documents)

print(f"{'CHECK':<34}{'BEFORE':>10}{'AFTER':>10}")
print("-" * 54)

checks = [
    ("paragraph breaks (want: many)", 0, cleaned_corpus.count("\n\n")),
    ("URLs embedded", 210, len(URL_PATTERN.findall(cleaned_corpus))),
    ("hyphen line-breaks", 124, len(HYPHEN_LINEBREAK.findall(cleaned_corpus))),
    ("footnote marks glued to words", 64,
     len(re.findall(r"[a-z]{3,}\.\d{1,2}\s+[A-Z]", cleaned_corpus))),
    ("'Source:' / 'Note:' lines", 84,
     len(re.findall(r"(?:^|\n)\s*(?:Sources?|Notes?)\s*:", cleaned_corpus))),
    ("rupee signs (want: UNCHANGED)", 200, cleaned_corpus.count("\u20b9")),
]
for label, before_value, after_value in checks:
    print(f"{label:<34}{before_value:>10,}{after_value:>10,}")

print("\nRunning heads remaining:")
for head in sorted(running_heads):
    print(f"   {cleaned_corpus.count(head):>4}   {head[:52]!r}")

# Confirm the paragraph markers survived and are now paragraph-initial.
paragraph_starts = [
    block for block in cleaned_corpus.split("\n\n")
    if re.match(r"^(6|7|8|9|10)\.\d{1,3}\s", block)
]
print(f"\nParagraphs beginning with a genuine marker: {len(paragraph_starts)}")

for sample_index in [0, 88]:
    raw_text = pages[sample_index].page_content
    clean_text = clean_page(raw_text, running_heads)

    print("=" * 78)
    print(f"PAGE INDEX {sample_index}   raw {len(raw_text):,} chars  ->  "
          f"clean {len(clean_text):,} chars")
    print("=" * 78)
    print("\n--- BEFORE ---")
    print(raw_text[:900])
    print("\n--- AFTER ---")
    print(clean_text[:900])
    print("\n")
