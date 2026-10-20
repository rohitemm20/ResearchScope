import json
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pdf_parser import parse_pdf
from phase2.routes import router as phase2_router
from phase2.topic_routes import router as topic_network_router


APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
OUTPUT_DIR = os.path.join(APP_DIR, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


app = FastAPI(title="ResearchScope - Phase 1 PDF Parser")


# Register existing Phase 2 routes
app.include_router(phase2_router)

# Register Research DNA topic-network routes
app.include_router(topic_network_router)


# Allow the local Vite dev server to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # ---- Validation ----------------------------------------------------

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted (content-type must be application/pdf).",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="File must have a .pdf extension.",
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File exceeds the 50MB limit.",
        )

    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="File does not look like a valid PDF.",
        )

    # ---- Save upload ---------------------------------------------------

    job_id = uuid.uuid4().hex[:12]

    saved_path = os.path.join(
        UPLOAD_DIR,
        f"{job_id}.pdf",
    )

    with open(saved_path, "wb") as f:
        f.write(contents)

    # ---- Parse ---------------------------------------------------------

    try:
        doc_result, full_text = parse_pdf(saved_path)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse PDF: {exc}",
        ) from exc

    doc_result["original_filename"] = file.filename
    doc_result["job_id"] = job_id

    # ---- Persist outputs -----------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.json",
    )

    txt_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.txt",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            doc_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # ---- Response ------------------------------------------------------

    return JSONResponse(
        {
            "job_id": job_id,
            "original_filename": file.filename,
            "num_pages": doc_result["num_pages"],
            "num_captions": len(doc_result["captions"]),
            "pages_summary": [
                {
                    "page_number": p["page_number"],
                    "layout": p["layout"],
                    "num_blocks": len(p["blocks"]),
                    "num_removed_header_footer": len(
                        p["removed_header_footer"]
                    ),
                }
                for p in doc_result["pages"]
            ],
            "json_download": f"/download/{job_id}/json",
            "txt_download": f"/download/{job_id}/txt",
            "result": doc_result,
        }
    )


@app.get("/download/{job_id}/{fmt}")
def download(job_id: str, fmt: str):
    if fmt not in ("json", "txt"):
        raise HTTPException(
            status_code=400,
            detail="fmt must be 'json' or 'txt'.",
        )

    path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.{fmt}",
    )

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="File not found.",
        )

    media_type = (
        "application/json"
        if fmt == "json"
        else "text/plain"
    )

    return FileResponse(
        path,
        media_type=media_type,
        filename=f"parsed_{job_id}.{fmt}",
    )
