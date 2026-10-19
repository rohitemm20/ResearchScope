"""
chunking.py
-----------
Stage: Redundancy Filter -> Smart Chunking

Greedy sentence packing that respects:
  * a hard token budget (max_tokens) so the summarizer never truncates,
  * section boundaries (starts a new chunk when the section changes and the
    current chunk is already big enough),
  * semantic breaks (starts a new chunk when the next sentence is far from
    the chunk's centroid).
"""
from typing import Callable

import numpy as np


def smart_chunks(sentences: list[dict], embeddings: np.ndarray,
                 count_tokens: Callable[[str], int],
                 max_tokens: int = 700, min_tokens: int = 200,
                 topic_break_sim: float = 0.30) -> list[dict]:
    chunks: list[dict] = []
    cur: list[int] = []
    cur_tokens = 0
    centroid = None

    def close():
        nonlocal cur, cur_tokens, centroid
        if not cur:
            return
        sents = [sentences[i] for i in cur]
        chunks.append({
            "chunk_id": len(chunks),
            "sentence_ids": [s["id"] for s in sents],
            "sections": list(dict.fromkeys(s["section"] for s in sents)),
            "pages": sorted({s["page"] for s in sents}),
            "n_tokens": cur_tokens,
            "text": " ".join(s["text"] for s in sents),
        })
        cur, cur_tokens, centroid = [], 0, None

    for i, s in enumerate(sentences):
        n = count_tokens(s["text"])
        if cur:
            prev = sentences[cur[-1]]
            sim = float(embeddings[i] @ centroid) / (np.linalg.norm(centroid) + 1e-9)
            if cur_tokens + n > max_tokens:
                close()
            elif cur_tokens >= min_tokens and (
                s["section"] != prev["section"] or sim < topic_break_sim
            ):
                close()
        cur.append(i)
        cur_tokens += n
        centroid = embeddings[i].copy() if centroid is None else centroid + embeddings[i]
    close()

    # merge a tiny trailing chunk into its predecessor if it fits
    if len(chunks) > 1 and chunks[-1]["n_tokens"] < min_tokens // 2 \
            and chunks[-2]["n_tokens"] + chunks[-1]["n_tokens"] <= max_tokens:
        last = chunks.pop()
        prev = chunks[-1]
        prev["sentence_ids"] += last["sentence_ids"]
        prev["sections"] = list(dict.fromkeys(prev["sections"] + last["sections"]))
        prev["pages"] = sorted(set(prev["pages"]) | set(last["pages"]))
        prev["n_tokens"] += last["n_tokens"]
        prev["text"] += " " + last["text"]
    return chunks
