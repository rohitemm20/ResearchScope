# ResearchScope — Phase 1: PDF Parser

Layout-aware PDF text extraction. This is **Phase 1 only**: upload → validate →
extract text blocks with coordinates → detect column layout → sort into
reading order → strip repeated headers/footers → separate captions →
save `parsed.json` + `parsed.txt`.

No section detection, summarization, or other NLP happens here — that's
later phases.

## Folder structure

```
researchscope-phase1/
├── backend/
│   ├── main.py            # FastAPI app: upload endpoint, validation, download endpoints
│   ├── pdf_parser.py       # Core parsing logic (PyMuPDF): columns, reading order, headers/footers, captions
│   ├── requirements.txt
│   ├── uploads/            # Uploaded PDFs land here (gitignored)
│   └── output/              # parsed .json and .txt land here (gitignored)
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx          # Upload UI + results view
        └── App.css
```

## Setup & run

### 1. Backend (FastAPI)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is now at `http://localhost:8000` (health check: `GET /health`).

### 2. Frontend (React + Vite)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, choose a PDF, click **Upload & Parse**.

> The frontend calls the backend at `http://localhost:8000` directly
> (hard-coded `API_BASE` in `App.jsx`) — CORS for `localhost:5173` is
> already enabled in `main.py`. Change both if you deploy elsewhere.

## How it works

**1. Upload & validation (`main.py`)**
The `/upload` endpoint checks the content-type is `application/pdf`, the
filename ends in `.pdf`, the file isn't empty or over 50MB, and the raw
bytes actually start with the `%PDF-` magic header. The file is saved to
`backend/uploads/<job_id>.pdf`.

**2. Block extraction (`extract_page_blocks`)**
For every page, PyMuPDF's `page.get_text("dict")` returns text grouped
into blocks → lines → spans, each with a bounding box `[x0, y0, x1, y1]`.
We keep only text blocks (image blocks are skipped in Phase 1) and join
each block's lines into a single string.

**3. Dynamic column detection (`detect_columns`)**
No coordinate is hard-coded. For each page we:
- Ignore very wide blocks (title, full-width captions) since they don't
  indicate column layout.
- Take the horizontal center of every remaining block, sort them, and
  find the single **largest gap** between consecutive centers.
- If that gap is wide enough (≥ 8% of the page width) and both sides
  have at least 2 blocks, the page is two-column and the boundary is
  placed in the middle of that gap. Otherwise, it's single-column.

This means a scanned two-column IEEE paper and a single-column PDF get
correctly classified without any per-document tuning, and the boundary
adapts to each page's actual margins and column widths.

**4. Reading order (`assign_reading_order`)**
Blocks are walked top-to-bottom. Full-width blocks (titles, section
headers, wide tables) flush whatever regular column blocks have been
accumulated so far — reading the left column top-to-bottom, then the
right column top-to-bottom — then the full-width block is emitted, and
accumulation resumes. This handles the common pattern of
`title → two columns → wide table/figure → two columns` correctly,
instead of assuming the whole page is one uninterrupted two-column block.

**5. Header/footer removal (`detect_headers_footers`)**
Blocks sitting in the top or bottom 12% of the page get their text
normalized (digits replaced with `#`, so "Page 3" and "Page 4" match) and
counted across all pages. Anything repeating on at least half the pages
(min 2) is treated as a running header/footer and removed from the body
— but kept in `removed_header_footer` per page in the JSON, in case you
want to inspect what was stripped.

**6. Caption detection (`CAPTION_REGEX`)**
Any remaining block whose text starts with `Figure`, `Fig.`, `Table`,
`Chart`, `Scheme`, `Algorithm`, or `Equation` followed by a number is
tagged `"type": "caption"` and kept in a separate `captions` list in the
JSON, rather than mixed into the main body text.

**7. Output**
- `output/<job_id>.json` — full structured result: per-page layout,
  per-block text/bbox/column/type/reading-order, removed headers/footers,
  and a document-level `captions` list.
- `output/<job_id>.txt` — the same content flattened into plain text, in
  reading order, page by page, with a `--- Captions ---` section per
  page when captions were found.

The `/upload` response also returns this JSON directly (under `"result"`)
plus download links, so the frontend doesn't need a second round trip.

## API summary

| Method | Path                      | Description                          |
|--------|---------------------------|----------------------------------------|
| GET    | `/health`                 | Liveness check                        |
| POST   | `/upload`                 | Upload a PDF, get parsed summary + result |
| GET    | `/download/{job_id}/json` | Download the full parsed JSON         |
| GET    | `/download/{job_id}/txt`  | Download the flattened parsed text    |
