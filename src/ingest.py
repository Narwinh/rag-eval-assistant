"""
Phase 1: Ingest & chunk the AWS Lambda Developer Guide.

The source PDF (data/raw/lambda-dg.pdf) is the full 3053-page Lambda Developer
Guide, which includes exhaustive per-runtime API references that are mostly
repetitive tables and not useful for a Q&A eval set. Instead of ingesting the
whole thing, we pull five coherent chapters (~299 printed pages) that cover
the concepts most relevant to actually using and reasoning about Lambda:
core concepts, the first-function tutorial, configuring functions, scaling/
concurrency, and permissions.

Page numbers below are PRINTED page numbers (as shown in the PDF's own
footers and table of contents). The PDF's internal 0-based page index is
always `printed_page + PAGE_OFFSET` for this document (verified by
cross-checking chapter headings against footers).
"""

import json
import re
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

PDF_PATH = Path("data/raw/lambda-dg.pdf")
OUTPUT_PATH = Path("data/processed/chunks.json")
PAGE_OFFSET = 13  # pdf_0based_index = printed_page + PAGE_OFFSET

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# (chapter, subsection, start_printed_page, end_printed_page)
SECTIONS = [
    # --- Core concepts (14-53) ---
    ("Core concepts", "Core concepts overview", 14, 18),
    ("Core concepts", "Running code", 19, 33),
    ("Core concepts", "Creating event-driven architectures", 34, 46),
    ("Core concepts", "Designing an application", 47, 53),
    # --- Create your first function (54-65) ---
    ("Create your first function", "Create your first function", 54, 59),
    ("Create your first function", "Invoke the function", 60, 63),
    ("Create your first function", "Clean up and next steps", 64, 65),
    # --- Configuring functions (762-910) ---
    ("Configuring functions", "Configuring functions overview", 762, 764),
    ("Configuring functions", ".zip file archives", 765, 777),
    ("Configuring functions", "Container images", 778, 787),
    ("Configuring functions", "Self-managed S3 code storage", 788, 793),
    ("Configuring functions", "Memory", 794, 796),
    ("Configuring functions", "Ephemeral storage", 797, 799),
    ("Configuring functions", "Instruction sets (ARM/x86)", 800, 803),
    ("Configuring functions", "Timeout", 804, 806),
    ("Configuring functions", "Environment variables", 807, 819),
    ("Configuring functions", "Attaching functions to a VPC", 820, 834),
    ("Configuring functions", "Attaching functions to resources in another account", 835, 841),
    ("Configuring functions", "Internet access for VPC functions", 842, 866),
    ("Configuring functions", "Inbound networking", 867, 870),
    ("Configuring functions", "File systems", 871, 878),
    ("Configuring functions", "Aliases", 879, 886),
    ("Configuring functions", "Versions", 887, 890),
    ("Configuring functions", "Tags", 891, 894),
    ("Configuring functions", "Response streaming", 895, 905),
    ("Configuring functions", "Metadata endpoint", 906, 910),
    # --- Function scaling (1027-1068) ---
    ("Function scaling", "Understanding and visualizing concurrency", 1027, 1032),
    ("Function scaling", "Calculating concurrency for a function", 1033, 1033),
    ("Function scaling", "Understanding reserved and provisioned concurrency", 1034, 1042),
    ("Function scaling", "Understanding concurrency and requests per second", 1043, 1043),
    ("Function scaling", "Concurrency quotas", 1044, 1046),
    ("Function scaling", "Configuring reserved concurrency", 1047, 1050),
    ("Function scaling", "Configuring provisioned concurrency", 1051, 1061),
    ("Function scaling", "Scaling behavior", 1062, 1063),
    ("Function scaling", "Monitoring concurrency", 1064, 1068),
    # --- Permissions (2224-2279) ---
    ("Permissions", "Execution role", 2224, 2237),
    ("Permissions", "Access permissions", 2238, 2279),
]

HEADER_LINE = "AWS Lambda Developer Guide"
FOOTER_RE = re.compile(r"^.{2,80}\s\d{1,4}$")  # "<running title> <page number>"

_encoding = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_encoding.encode(text))


def clean_page_text(text: str) -> str:
    """Strip the repeated running header and page-number footer line."""
    lines = [l.strip() for l in text.split("\n")]
    cleaned = []
    for line in lines:
        if not line:
            continue
        if line == HEADER_LINE:
            continue
        if FOOTER_RE.match(line) and len(line.split()) <= 8:
            # Likely a "<Section Name> <page#>" footer; drop it.
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_section_text(reader: PdfReader, start_page: int, end_page: int) -> tuple[str, list[str]]:
    """Extract and clean text for a printed page range. Returns (text, failures)."""
    failures = []
    parts = []
    for printed_page in range(start_page, end_page + 1):
        idx = printed_page + PAGE_OFFSET
        try:
            raw = reader.pages[idx].extract_text()
        except Exception as e:
            failures.append(f"printed page {printed_page} (pdf index {idx}): {e}")
            continue
        if not raw or not raw.strip():
            failures.append(f"printed page {printed_page} (pdf index {idx}): empty extraction")
            continue
        parts.append(clean_page_text(raw))
    return "\n".join(parts), failures


def chunk_documents():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Expected source PDF at {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    all_failures = []
    chunk_counter = 0

    for chapter, subsection, start_page, end_page in SECTIONS:
        text, failures = extract_section_text(reader, start_page, end_page)
        all_failures.extend(failures)

        if not text.strip():
            print(f"  [WARN] No text extracted for '{chapter} > {subsection}'")
            continue

        pieces = splitter.split_text(text)
        for piece in pieces:
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": f"c{chunk_counter:04d}",
                "text": piece,
                "chapter": chapter,
                "subsection": subsection,
                "source_doc": f"{chapter} > {subsection}",
                "page_range": [start_page, end_page],
                "token_count": _token_len(piece),
            })

    return all_chunks, all_failures


def main():
    print(f"Loading PDF: {PDF_PATH}")
    chunks, failures = chunk_documents()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    total_pages = sum(end - start + 1 for _, _, start, end in SECTIONS)
    avg_tokens = sum(c["token_count"] for c in chunks) / len(chunks) if chunks else 0

    print("\n--- Ingest summary ---")
    print(f"Chapters/sections processed : {len(SECTIONS)}")
    print(f"Printed pages covered        : {total_pages}")
    print(f"Chunks produced               : {len(chunks)}")
    print(f"Average chunk size (tokens)   : {avg_tokens:.1f}")
    print(f"Parsing failures               : {len(failures)}")
    if failures:
        print("First few failures:")
        for f_ in failures[:10]:
            print(f"  - {f_}")
    print(f"Wrote chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
