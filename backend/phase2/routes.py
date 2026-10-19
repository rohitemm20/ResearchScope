"""
routes.py - FastAPI router for Phase 2. Add to backend/main.py:

    from phase2.routes import router as phase2_router
    app.include_router(phase2_router)
"""

import json
import os
import re

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .pipeline import run_pipeline, semantic_search
from .text_cleaning import load_parsed_json
from .relevance import check_relevance, clean_interests


router = APIRouter(prefix="/phase2", tags=["phase2"])

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BACKEND_DIR, "output")
JOB_RE = re.compile(r"^[0-9a-f]{12}$")


class Interests(BaseModel):
    domain: str
    subdomains: list[str] = []


def _check(job_id: str):
    if not JOB_RE.match(job_id):
        raise HTTPException(400, "Invalid job_id.")


@router.post("/process/{job_id}")
def process(
    job_id: str,
    quality: str = Query("fast", pattern="^(fast|best)$"),
    redundancy_threshold: float = Query(0.85, ge=0.5, le=0.99)
):
    """Run summarise + keywords + index on an already-parsed job."""
    _check(job_id)

    path = os.path.join(OUTPUT_DIR, f"{job_id}.json")

    if not os.path.exists(path):
        raise HTTPException(
            404,
            "Parsed JSON not found. Upload the PDF first."
        )

    try:
        return run_pipeline(
            load_parsed_json(path),
            job_id,
            OUTPUT_DIR,
            quality=quality,
            redundancy_threshold=redundancy_threshold
        )

    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/result/{job_id}")
def result(job_id: str):
    _check(job_id)

    path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}_phase2.json"
    )

    if not os.path.exists(path):
        raise HTTPException(
            404,
            "Not processed yet. POST /phase2/process/{job_id}."
        )

    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/search/{job_id}")
def search(
    job_id: str,
    q: str = Query(..., min_length=2),
    k: int = Query(5, ge=1, le=20)
):
    _check(job_id)

    if not os.path.exists(
        os.path.join(OUTPUT_DIR, f"{job_id}.faiss")
    ):
        raise HTTPException(
            404,
            "Index not built. POST /phase2/process/{job_id} first."
        )

    return {
        "query": q,
        "results": semantic_search(
            job_id,
            OUTPUT_DIR,
            q,
            k
        )
    }


@router.post("/relevance/{job_id}")
def relevance(
    job_id: str,
    interests: Interests
):
    """
    Calculate how relevant a processed research paper is
    to the user's selected domain and subdomains.
    """

    _check(job_id)

    phase2_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}_phase2.json"
    )

    index_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.faiss"
    )

    if not os.path.exists(phase2_path):
        raise HTTPException(
            404,
            "Phase 2 result not found. Process the paper first."
        )

    if not os.path.exists(index_path):
        raise HTTPException(
            404,
            "Semantic index not found. Process the paper first."
        )

    domain, subdomains = clean_interests(
        interests.domain,
        interests.subdomains
    )

    if not domain:
        raise HTTPException(
            422,
            "Domain cannot be empty."
        )

    try:
        return check_relevance(
            job_id,
            OUTPUT_DIR,
            domain,
            subdomains
        )

    except ValueError as exc:
        raise HTTPException(
            422,
            str(exc)
        ) from exc