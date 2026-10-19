"""
redundancy.py
-------------
Branch B: Embeddings -> Redundancy Filter

Greedy, order-preserving: a sentence is kept only if its cosine similarity
to every already-kept sentence is below `threshold`. Earlier sentences win,
so the abstract/intro wording survives and later restatements are dropped.
"""
import numpy as np


def filter_redundant(embeddings: np.ndarray, sentences: list[dict],
                     threshold: float = 0.85):
    """returns (kept_indices, removed=[{id, text, similar_to, similarity}])"""
    kept: list[int] = []
    removed: list[dict] = []
    kept_mat = np.empty((0, embeddings.shape[1]), dtype="float32")

    for i, vec in enumerate(embeddings):
        if kept:
            sims = kept_mat @ vec
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                removed.append({
                    "id": i, "text": sentences[i]["text"],
                    "similar_to": kept[j],
                    "similarity": round(float(sims[j]), 4),
                })
                continue
        kept.append(i)
        kept_mat = np.vstack([kept_mat, vec])
    return kept, removed
