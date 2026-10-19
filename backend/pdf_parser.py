
"""
pdf_parser.py
-------------

Phase 1 PDF parsing engine for ResearchScope.

Pipeline (per document):

1. Extract text blocks + coordinates from every page (PyMuPDF).
2. Detect whether a page is single-column or two-column without any
   hard-coded x-split.
3. Sort blocks into human reading order.
4. Detect repeated headers/footers across the document and remove them.
5. Detect figure/table captions and keep them separately.
6. Return a structured dictionary and flattened reading-order text.

No section detection, summarization, or other NLP happens here.
"""

import os
import re
from collections import Counter

import pymupdf


# Matches captions such as:
# Figure 1:
# Fig. 2 -
# Table 3.
# Chart 4
CAPTION_REGEX = re.compile(
    r"^(figure|fig\.?|table|chart|scheme|algorithm|equation|eq\.?)\s*\d+",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Step 1: Extract raw text blocks with coordinates
# --------------------------------------------------------------------------

def extract_page_blocks(page):
    """
    Return a list of dictionaries containing text and bounding boxes.

    Each block has:
        {
            "text": "...",
            "bbox": [x0, y0, x1, y1]
        }

    PyMuPDF's dictionary representation groups text into:
        blocks -> lines -> spans

    Image blocks are skipped because Phase 1 only handles text.
    """

    raw = page.get_text("dict")

    blocks = []

    for block in raw.get("blocks", []):

        # Type 0 = text block
        # Type 1 = image block
        if block.get("type") != 0:
            continue

        line_texts = []

        for line in block.get("lines", []):

            line_text = "".join(
                span.get("text", "")
                for span in line.get("spans", [])
            )

            if line_text.strip():
                line_texts.append(line_text)

        text = "\n".join(line_texts).strip()

        if not text:
            continue

        blocks.append(
            {
                "text": text,
                "bbox": list(block["bbox"]),
            }
        )

    return blocks


# --------------------------------------------------------------------------
# Step 2: Dynamic column detection
# --------------------------------------------------------------------------

def detect_columns(blocks, page_width):
    """
    Determine whether a page is single-column or two-column.

    No fixed x-coordinate is used.

    The algorithm:

    1. Ignore very wide blocks because they are likely full-width
       titles, captions, or section headers.
    2. Calculate the horizontal center of each remaining block.
    3. Sort the centers.
    4. Find the largest horizontal gap.
    5. If the gap is large enough and both sides contain enough blocks,
       classify the page as two-column.
    """

    span_threshold = 0.6 * page_width

    candidates = [
        block
        for block in blocks
        if (block["bbox"][2] - block["bbox"][0]) < span_threshold
    ]

    if len(candidates) < 4:
        return False, None

    centers = sorted(
        (block["bbox"][0] + block["bbox"][2]) / 2.0
        for block in candidates
    )

    best_gap = 0.0
    best_idx = -1

    for i in range(1, len(centers)):
        gap = centers[i] - centers[i - 1]

        if gap > best_gap:
            best_gap = gap
            best_idx = i

    # Gap must be at least 8% of page width.
    if best_idx == -1 or best_gap < 0.08 * page_width:
        return False, None

    left_cluster = centers[:best_idx]
    right_cluster = centers[best_idx:]

    if len(left_cluster) < 2 or len(right_cluster) < 2:
        return False, None

    boundary = (
        left_cluster[-1] + right_cluster[0]
    ) / 2.0

    return True, boundary


# --------------------------------------------------------------------------
# Step 3: Reading-order assignment
# --------------------------------------------------------------------------

def assign_reading_order(
    blocks,
    page_width,
    is_two_col,
    boundary,
):
    """
    Sort blocks into human reading order.

    For two-column pages:

        Left column top -> bottom
        then
        Right column top -> bottom

    Full-width blocks are emitted at their vertical position so that
    structures such as:

        Title
        Abstract
        Left/right body
        Full-width figure
        Left/right body

    remain approximately in the correct order.
    """

    span_threshold = 0.6 * page_width

    # Add temporary geometry information.
    for block in blocks:
        block["_width"] = (
            block["bbox"][2] - block["bbox"][0]
        )

        block["_cx"] = (
            block["bbox"][0] + block["bbox"][2]
        ) / 2.0

        block["_y0"] = block["bbox"][1]

    by_y = sorted(
        blocks,
        key=lambda block: block["_y0"]
    )

    def flush(buffer):
        if not buffer:
            return []

        if is_two_col:

            left = sorted(
                [
                    block
                    for block in buffer
                    if block["_cx"] < boundary
                ],
                key=lambda block: block["_y0"],
            )

            right = sorted(
                [
                    block
                    for block in buffer
                    if block["_cx"] >= boundary
                ],
                key=lambda block: block["_y0"],
            )

            return left + right

        return sorted(
            buffer,
            key=lambda block: block["_y0"],
        )

    ordered = []
    buffer = []

    for block in by_y:

        if (
            is_two_col
            and block["_width"] >= span_threshold
        ):
            ordered.extend(flush(buffer))
            buffer = []

            block["column"] = "full"

            ordered.append(block)

        else:
            buffer.append(block)

    ordered.extend(flush(buffer))

    # Assign final column labels.
    for block in ordered:

        if "column" not in block:

            if is_two_col:
                block["column"] = (
                    "left"
                    if block["_cx"] < boundary
                    else "right"
                )
            else:
                block["column"] = "single"

    return ordered


# --------------------------------------------------------------------------
# Step 4: Text normalization
# --------------------------------------------------------------------------

def normalize_text(text):
    """
    Normalize text for header/footer comparison.

    Example:

        Page 1
        Page 2
        Page 3

    becomes:

        page #

    This allows page numbers to be detected as the same repeated
    header/footer.
    """

    text = re.sub(
        r"\d+",
        "#",
        text.strip(),
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.lower()


# --------------------------------------------------------------------------
# Step 5: Header/footer detection
# --------------------------------------------------------------------------

def detect_headers_footers(
    pages_blocks,
    page_height,
    min_fraction=0.5,
):
    """
    Detect repeated headers and footers.

    A block qualifies if:

    - it is located in the top 12% or bottom 12% of the page
    - its normalized text appears on at least 50% of pages
    - at least two pages exist

    This prevents a one-page paper from incorrectly removing
    legitimate content.
    """

    top_margin = 0.12 * page_height
    bottom_margin = 0.88 * page_height

    counter = Counter()

    for blocks in pages_blocks:

        seen_on_this_page = set()

        for block in blocks:

            y0 = block["bbox"][1]
            y1 = block["bbox"][3]

            is_top = y1 < top_margin
            is_bottom = y0 > bottom_margin

            if not (is_top or is_bottom):
                continue

            normalized = normalize_text(
                block["text"]
            )

            if (
                normalized
                and normalized not in seen_on_this_page
            ):
                counter[normalized] += 1
                seen_on_this_page.add(normalized)

    number_of_pages = len(pages_blocks)

    threshold = max(
        2,
        int(min_fraction * number_of_pages),
    )

    return {
        text
        for text, count in counter.items()
        if count >= threshold
    }


# --------------------------------------------------------------------------
# Step 6: Main PDF parser
# --------------------------------------------------------------------------

def parse_pdf(path):
    """
    Parse a PDF end-to-end.

    Returns:

        (
            result_dict,
            full_text
        )
    """

    doc = pymupdf.open(path)

    raw_pages = []

    for page in doc:

        raw_pages.append(
            {
                "blocks": extract_page_blocks(page),
                "width": page.rect.width,
                "height": page.rect.height,
            }
        )

    doc.close()

    # Use the first page height as the reference height.
    page_height_ref = (
        raw_pages[0]["height"]
        if raw_pages
        else 792.0
    )

    repeated_texts = detect_headers_footers(
        [page["blocks"] for page in raw_pages],
        page_height_ref,
    )

    result_pages = []
    text_chunks = []
    all_captions = []

    for page_index, page_data in enumerate(raw_pages):

        page_width = page_data["width"]

        kept_blocks = []
        removed_blocks = []

        # --------------------------------------------------------------
        # Remove repeated headers/footers
        # --------------------------------------------------------------

        for block in page_data["blocks"]:

            if (
                normalize_text(block["text"])
                in repeated_texts
            ):
                removed_blocks.append(
                    block["text"]
                )
            else:
                kept_blocks.append(block)

        # --------------------------------------------------------------
        # Detect layout
        # --------------------------------------------------------------

        is_two_col, boundary = detect_columns(
            kept_blocks,
            page_width,
        )

        # --------------------------------------------------------------
        # Sort reading order
        # --------------------------------------------------------------

        ordered = assign_reading_order(
            kept_blocks,
            page_width,
            is_two_col,
            boundary,
        )

        block_records = []
        body_texts = []
        caption_texts = []

        # --------------------------------------------------------------
        # Build block records
        # --------------------------------------------------------------

        for block_index, block in enumerate(ordered):

            is_caption = bool(
                CAPTION_REGEX.match(
                    block["text"].strip()
                )
            )

            record = {
                "id": block_index,
                "type": (
                    "caption"
                    if is_caption
                    else "body"
                ),
                "text": block["text"],
                "bbox": [
                    round(value, 2)
                    for value in block["bbox"]
                ],
                "column": block["column"],
                "reading_order": block_index,
            }

            block_records.append(record)

            if is_caption:

                caption_texts.append(
                    block["text"]
                )

                all_captions.append(
                    {
                        "page": page_index + 1,
                        "text": block["text"],
                    }
                )

            else:

                body_texts.append(
                    block["text"]
                )

        # --------------------------------------------------------------
        # Store page result
        # --------------------------------------------------------------

        result_pages.append(
            {
                "page_number": page_index + 1,
                "layout": (
                    "two-column"
                    if is_two_col
                    else "single-column"
                ),
                "column_boundary_x": (
                    round(boundary, 2)
                    if boundary is not None
                    else None
                ),
                "blocks": block_records,
                "removed_header_footer": removed_blocks,
            }
        )

        # --------------------------------------------------------------
        # Build flattened text
        # --------------------------------------------------------------

        layout_name = (
            "two-column"
            if is_two_col
            else "single-column"
        )

        page_header = (
            f"\n\n===== Page "
            f"{page_index + 1} "
            f"({layout_name}) =====\n"
        )

        text_chunks.append(page_header)

        if body_texts:
            text_chunks.append(
                "\n\n".join(body_texts)
            )

        if caption_texts:
            text_chunks.append(
                "\n\n--- Captions ---\n"
                + "\n".join(caption_texts)
            )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    result = {
        "filename": os.path.basename(path),
        "num_pages": len(raw_pages),
        "pages": result_pages,
        "captions": all_captions,
    }

    full_text = (
        "".join(text_chunks).strip()
        + "\n"
    )

    return result, full_text
