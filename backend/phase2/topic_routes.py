"""Router for Research DNA graph endpoints."""
import json
import logging
import os
import re

from fastapi import APIRouter, HTTPException, Query

from .text_cleaning import load_parsed_json
from .topic_network import METHOD_NAME, build_topic_network

router = APIRouter(prefix="/phase2/topic-network", tags=["research-dna"])
log = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BACKEND_DIR, "output")
JOB_RE = re.compile(r"^[0-9a-f]{12}$")


def _cache_path(job_id: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{job_id}_topic_network.json")


def _read_cache(cache: str, source: str):
    """Use the cached graph only if it is newer than the parsed PDF and was
    produced by the current algorithm version."""
    try:
        if os.path.getmtime(cache) < os.path.getmtime(source):
            return None
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("method", {}).get("name") != METHOD_NAME:
            return None
        return data
    except (OSError, ValueError):
        return None


@router.get("/{job_id}")
def topic_network(job_id: str, refresh: bool = Query(False)):
    if not JOB_RE.fullmatch(job_id):
        raise HTTPException(400, "Invalid job_id.")

    path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Parsed PDF not found. Upload the paper first.")

    cache = _cache_path(job_id)
    if not refresh:
        cached = _read_cache(cache, path)
        if cached is not None:
            return cached

    try:
        parsed = load_parsed_json(path)
        parsed["job_id"] = job_id
        graph = build_topic_network(parsed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        # Do not expose stack traces or local paths to clients.
        log.exception("Topic network failed for job %s", job_id)
        raise HTTPException(
            500, "Topic network could not be generated. Check backend logs."
        ) from exc

    try:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False)
    except OSError:
        log.warning("Could not write topic network cache for %s", job_id)
    return graph
