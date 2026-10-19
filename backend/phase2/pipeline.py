"""
pipeline.py
-----------
Orchestrates the whole Phase 2 flow:

  parsed JSON -> clean -> spaCy sentences -> embeddings
                                   |-> FAISS index (semantic search)
                                   '-> redundancy filter -> smart chunks
                                         -> summarizer -> final summary
                                         -> KeyBERT keywords
Models are loaded lazily once and re-used (they are slow to load).
"""
import json
import os
import time

from .text_cleaning import clean_document, load_parsed_json
from .nlp_utils import split_sentences
from .search import SemanticIndex
from .redundancy import filter_redundant
from .chunking import smart_chunks

_embedder = None
_summarizers: dict = {}


def get_embedder():
    global _embedder
    if _embedder is None:
        from .embeddings import Embedder
        _embedder = Embedder()
    return _embedder


def get_summarizer(quality: str = "fast"):
    if quality not in _summarizers:
        from .summarizer import MODELS, Summarizer
        _summarizers[quality] = Summarizer(MODELS[quality])
    return _summarizers[quality]


def run_pipeline(parsed: dict, job_id: str, out_dir: str,
                 quality: str = "fast", redundancy_threshold: float = 0.85,
                 top_keywords: int = 10) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t = {}
    clock = time.time()

    def tick(name):
        nonlocal clock
        now = time.time(); t[name] = round(now - clock, 2); clock = now

    # 1. cleaning ---------------------------------------------------------
    paragraphs = clean_document(parsed)
    tick("cleaning")

    # 2. spaCy sentences + tokens -------------------------------------------
    sentences = split_sentences(paragraphs)
    if not sentences:
        raise ValueError("No usable sentences after cleaning.")
    tick("spacy")

    # 3. embeddings -----------------------------------------------------------
    emb = get_embedder()
    vectors = emb.encode([s["text"] for s in sentences])
    tick("embeddings")

    # 4a. FAISS index over ALL sentences (search should see everything) ----
    SemanticIndex(vectors, sentences).save(out_dir, job_id)
    tick("faiss")

    # 4b. redundancy filter -> smart chunking ------------------------------
    kept, removed = filter_redundant(vectors, sentences, redundancy_threshold)
    kept_sents = [dict(sentences[i], id=n) for n, i in enumerate(kept)]
    # keep original ids available for traceability
    for n, i in enumerate(kept):
        kept_sents[n]["orig_id"] = sentences[i]["id"]
    kept_vecs = vectors[kept]
    tick("redundancy")

    summ = get_summarizer(quality)
    chunks = smart_chunks(kept_sents, kept_vecs, summ.count_tokens)
    tick("chunking")

    # 5. summarisation ---------------------------------------------------------
    chunks = summ.summarize_chunks(chunks)
    final = summ.final_summary([c["summary"] for c in chunks])
    tick("summarisation")

    # 6. keywords (from the summary; document-level as a second view) ----
    from .keywords import extract_keywords
    kw_summary = extract_keywords(final, emb.model, top_n=top_keywords)
    kw_doc = extract_keywords(" ".join(s["text"] for s in kept_sents),
                              emb.model, top_n=top_keywords)
    tick("keywords")

    result = {
        "job_id": job_id,
        "title_source": parsed.get("original_filename"),
        "stats": {
            "paragraphs": len(paragraphs),
            "sentences": len(sentences),
            "after_redundancy_filter": len(kept),
            "removed_redundant": len(removed),
            "chunks": len(chunks),
            "summarizer_model": quality,
        },
        "summary": final,
        "keywords": kw_summary,
        "keywords_document": kw_doc,
        "chunks": chunks,
        "removed_redundant": removed,
        "timings_sec": t,
    }
    with open(os.path.join(out_dir, f"{job_id}_phase2.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def semantic_search(job_id: str, out_dir: str, query: str, k: int = 5) -> list[dict]:
    index = SemanticIndex.load(out_dir, job_id)
    qvec = get_embedder().encode([query])[0]
    return index.search(qvec, k)
