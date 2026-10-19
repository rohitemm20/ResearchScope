"""
text_cleaning.py
----------------
Stage: JSON Output -> Text Cleaning

Input:
    Parsed JSON produced by pdf_parser.parse_pdf()

Output:
    A list of paragraphs:
        {
            "page": ...,
            "section": ...,
            "text": ...,
            "block_ids": [...]
        }

Cleaning pipeline:
  1. Walk pages/blocks in reading order.
  2. Skip captions.
  3. Track the current section from heading blocks.
  4. Stop at REFERENCES.
  5. Remove obvious noise/table fragments.
  6. Normalize Unicode and PDF line wrapping.
  7. Remove citation markers.
  8. Merge paragraphs split across columns/pages.
"""

import json
import re
import unicodedata


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------

HEADING_RE = re.compile(
    r"^(?:(?:[IVX]+|[A-Z]|\d+(?:\.\d+)*)\.)?\s*"
    r"[A-Za-z][^\n]{0,80}$"
)

REFERENCES_RE = re.compile(
    r"^\s*(?:[IVX]+\.\s*)?references\s*$",
    re.I
)

REF_ENTRY_RE = re.compile(
    r"^\s*\[\d+\]\s*"
)

CITATION_RE = re.compile(
    r"\s*\[\s*\d+(?:\s*[-–—,]\s*\d+)*\s*\]"
)

HYPHEN_BREAK_RE = re.compile(
    r"(\w+)-\s*\n\s*(\w+)"
)

BULLET_RE = re.compile(
    r"[\u25cf\u2022\u25aa\u25a0\u25e6]"
)

ABSTRACT_RE = re.compile(
    r"^\s*abstract\s*[\u2014\u2013:.\-]?\s*",
    re.I
)

INDEX_TERMS_RE = re.compile(
    r"^\s*index terms\s*[\u2014\u2013:.\-]",
    re.I
)


# Prefixes whose line-end hyphen is a real hyphen.
KEEP_HYPHEN_PREFIXES = {
    "ride", "cross", "non", "multi", "real", "time",
    "spatio", "two", "self", "well", "high", "low",
    "data", "price", "best", "supply", "game", "state",
    "long", "short", "deep", "semi", "peer", "end",
    "least", "mean", "demand", "market", "profit",
    "sub", "inter",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normalize Unicode and remove problematic PDF characters."""

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Soft hyphen
    text = text.replace("\u00ad", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null/control characters while preserving normal punctuation
    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        " ",
        text,
    )

    return text


def dehyphenate(text: str) -> str:

    def _join(match):
        left = match.group(1)
        right = match.group(2)

        if (
            left.lower() in KEEP_HYPHEN_PREFIXES
            or right[:1].isupper()
        ):
            return f"{left}-{right}"

        return f"{left}{right}"

    return HYPHEN_BREAK_RE.sub(_join, text)


def clean_block_text(text: str) -> str:

    if not text:
        return ""

    text = normalize_unicode(text)
    text = dehyphenate(text)

    # Remove citations such as [7], [11], [11-15]
    text = CITATION_RE.sub("", text)

    # PDF wrapped lines -> spaces
    text = re.sub(r"\s*\n\s*", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove whitespace before punctuation
    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    # Fix en/em dash spacing
    text = re.sub(
        r"(\w)([\u2013\u2014])\s+(\w)",
        r"\1\2\3",
        text,
    )

    return text.strip()


def alpha_ratio(text: str) -> float:

    if not text:
        return 0.0

    letters = sum(c.isalpha() for c in text)

    return letters / max(len(text), 1)


def is_heading(raw: str) -> bool:

    if not raw:
        return False

    t = raw.strip()

    # Headings should normally be short single-line blocks
    if "\n" in t:
        return False

    if len(t) > 90:
        return False

    if REFERENCES_RE.match(t):
        return True

    # Numbered headings:
    # 1. Introduction
    # 2.1 Nash Equilibrium
    # 3.2.1 Mixed Strategies
    if re.match(
        r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]",
        t,
    ):
        return True

    # Roman numeral headings:
    # I. Introduction
    # II. Results
    if re.match(
        r"^[IVX]+\.?\s+[A-Za-z]",
        t,
    ):
        return True

    return False


def is_noise(raw: str, drop_bullets: bool) -> bool:

    if not raw:
        return True

    t = raw.strip()

    if not t:
        return True

    # Reference entries
    if REF_ENTRY_RE.match(t):
        return True

    # Table bullets
    if drop_bullets and BULLET_RE.search(t):
        return True

    # Very small fragments
    if len(t) < 3:
        return True

    # Only classify SHORT fragments as possible equation/number noise.
    #
    # Important:
    # We do NOT discard long scientific sentences just because their
    # alpha ratio is relatively low.
    if len(t) < 60 and alpha_ratio(t) < 0.35:
        return True

    return False


def ends_sentence(text: str) -> bool:

    if not text:
        return False

    return bool(
        re.search(
            r"""[.!?:][\]"'\u201d\u2019)]*$""",
            text.strip(),
        )
    )


# ----------------------------------------------------------------------
# Main API
# ----------------------------------------------------------------------

def load_parsed_json(path: str) -> dict:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _table_start_y(page: dict):

    ys = [
        b["bbox"][1]
        for b in page.get("blocks", [])
        if (
            b.get("type") == "caption"
            and b.get("text", "")
            .strip()
            .lower()
            .startswith("table")
        )
    ]

    return min(ys) if ys else None


def _looks_like_table_cell(raw: str) -> bool:

    if not raw:
        return True

    t = raw.strip()

    # Only treat very short fragments as table cells.
    #
    # Previously this rule was too aggressive and could remove
    # legitimate prose from pages containing tables.
    return (
        len(t) < 120
        and not re.search(
            r"[.!?]\s*$",
            t,
        )
        and alpha_ratio(t) < 0.75
    )


def clean_document(
    parsed: dict,
    drop_bullets: bool = True,
    min_paragraph_chars: int = 20,
    include_front_matter: bool = False,
) -> list[dict]:
    """
    Return cleaned paragraphs in reading order.
    """

    paragraphs: list[dict] = []

    section = "Front Matter"
    in_references = False

    # Debug counters
    stats = {
        "total_blocks": 0,
        "captions_removed": 0,
        "table_cells_removed": 0,
        "headings_found": 0,
        "noise_removed": 0,
        "empty_removed": 0,
        "paragraphs_before_merge": 0,
        "paragraphs_after_merge": 0,
    }

    for page in parsed.get("pages", []):

        table_y = _table_start_y(page)

        for block in page.get("blocks", []):

            stats["total_blocks"] += 1

            raw = block.get("text", "")

            if not raw:
                stats["empty_removed"] += 1
                continue

            # ----------------------------------------------------------
            # Table cells
            # ----------------------------------------------------------

            if (
                table_y is not None
                and block.get("bbox")
                and block["bbox"][1] >= table_y
                and block.get("type") != "caption"
                and not is_heading(raw)
                and _looks_like_table_cell(raw)
            ):
                stats["table_cells_removed"] += 1
                continue

            # ----------------------------------------------------------
            # Captions
            # ----------------------------------------------------------

            if block.get("type") == "caption":
                stats["captions_removed"] += 1
                continue

            # ----------------------------------------------------------
            # References
            # ----------------------------------------------------------

            first_line = raw.strip().split("\n")[0]

            if (
                REFERENCES_RE.match(first_line)
                and len(raw.strip()) < 60
            ):
                in_references = True
                break

            # ----------------------------------------------------------
            # Headings
            # ----------------------------------------------------------

            if is_heading(raw):

                stats["headings_found"] += 1

                section = clean_block_text(raw)

                # Keep heading information for section tracking,
                # but it will be removed from the sentence corpus.
                paragraphs.append({
                    "page": page["page_number"],
                    "section": section,
                    "text": section,
                    "block_ids": [block["id"]],
                    "is_heading": True,
                })

                continue

            # ----------------------------------------------------------
            # Obvious noise
            # ----------------------------------------------------------

            if is_noise(
                raw,
                drop_bullets,
            ):
                stats["noise_removed"] += 1
                continue

            # ----------------------------------------------------------
            # Clean actual prose
            # ----------------------------------------------------------

            text = clean_block_text(raw)

            if not text:
                stats["empty_removed"] += 1
                continue

            # Index terms are not part of summary prose
            if INDEX_TERMS_RE.match(text):
                continue

            # Abstract label
            if ABSTRACT_RE.match(text):

                text = ABSTRACT_RE.sub(
                    "",
                    text,
                ).strip()

                section = "Abstract"

            if not text:
                continue

            # ----------------------------------------------------------
            # Keep paragraph
            # ----------------------------------------------------------

            paragraphs.append({
                "page": page["page_number"],
                "section": section,
                "text": text,
                "block_ids": [block["id"]],
                "is_heading": False,
            })

        if in_references:
            break

    stats["paragraphs_before_merge"] = len(paragraphs)

    # Remove front matter only when we actually have other content.
    if not include_front_matter:

        non_front = [
            p
            for p in paragraphs
            if p["section"] != "Front Matter"
        ]

        if non_front:
            paragraphs = non_front

    paragraphs = _merge_split_paragraphs(
        paragraphs,
        min_paragraph_chars,
    )

    stats["paragraphs_after_merge"] = len(paragraphs)

    # --------------------------------------------------------------
    # Debug information
    # --------------------------------------------------------------

    print("\n========== TEXT CLEANING ==========")
    print("Total blocks:", stats["total_blocks"])
    print("Captions removed:", stats["captions_removed"])
    print("Table cells removed:", stats["table_cells_removed"])
    print("Headings detected:", stats["headings_found"])
    print("Noise removed:", stats["noise_removed"])
    print("Empty blocks removed:", stats["empty_removed"])
    print("Paragraphs before merge:", stats["paragraphs_before_merge"])
    print("Paragraphs after merge:", stats["paragraphs_after_merge"])

    if paragraphs:
        print(
            "First cleaned paragraph:",
            repr(paragraphs[0]["text"][:300]),
        )
    else:
        print("WARNING: ZERO PARAGRAPHS AFTER CLEANING")

    print("===================================\n")

    return paragraphs


def _merge_split_paragraphs(
    paras: list[dict],
    min_chars: int,
) -> list[dict]:
    """
    Join a paragraph that was cut by a column/page break.
    """

    merged: list[dict] = []

    for p in paras:

        prev = merged[-1] if merged else None

        if (
            prev
            and not p["is_heading"]
            and not prev["is_heading"]
            and not ends_sentence(prev["text"])
            and p["text"][:1].islower()
        ):

            if prev["text"].endswith("-"):

                stem = re.search(
                    r"(\w+)-$",
                    prev["text"],
                )

                keep = (
                    stem
                    and stem.group(1).lower()
                    in KEEP_HYPHEN_PREFIXES
                )

                if not keep:
                    prev["text"] = prev["text"][:-1]

                prev["text"] += p["text"]

            else:
                prev["text"] += " " + p["text"]

            prev["block_ids"] += p["block_ids"]

        else:
            merged.append(dict(p))

    # Remove headings from the actual sentence corpus,
    # but preserve their section information in body paragraphs.
    result = [
        m
        for m in merged
        if (
            not m["is_heading"]
            and len(m["text"].strip()) >= min_chars
        )
    ]

    return result


def paragraphs_to_text(
    paras: list[dict],
) -> str:

    return "\n\n".join(
        p["text"]
        for p in paras
    )


# ----------------------------------------------------------------------
# Direct test
# ----------------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m phase2.text_cleaning "
            "<parsed_json>"
        )
        sys.exit(1)

    parsed = load_parsed_json(
        sys.argv[1]
    )

    paragraphs = clean_document(
        parsed
    )

    for p in paragraphs:
        print(
            f"[p{p['page']} | {p['section']}] "
            f"{p['text'][:160]}\n"
        )