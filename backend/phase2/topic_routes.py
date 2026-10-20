"""Optional router for Research DNA graph endpoints."""
import os
import re

from fastapi import APIRouter, HTTPException

from .text_cleaning import load_parsed_json
from .topic_network import build_topic_network

router = APIRouter(prefix="/phase2/topic-network", tags=["research-dna"])
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BACKEND_DIR, "output")
JOB_RE = re.compile(r"^[0-9a-f]{12}$")

@router.get("/{job_id}")
def topic_network(job_id: str):
    if not JOB_RE.fullmatch(job_id):
        raise HTTPException(400, "Invalid job_id.")
    path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Parsed PDF not found. Upload the paper first.")
    try:
        parsed = load_parsed_json(path)
        parsed["job_id"] = job_id
        return build_topic_network(parsed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        # Avoid exposing stack traces or local paths to clients.
        raise HTTPException(500, "Topic network could not be generated. Check backend logs.") from exc
