"""
search.py
---------
Branch A: Embeddings -> Semantic Search (FAISS)

IndexFlatIP over normalised vectors = exact cosine search. For a single
paper (a few hundred sentences) exact search is instant; no ANN needed.
"""
import json
import os

import numpy as np


class SemanticIndex:
    def __init__(self, embeddings: np.ndarray, sentences: list[dict]):
        import faiss
        self.sentences = sentences
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vec.reshape(1, -1), k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            s = self.sentences[idx]
            out.append({"id": int(idx), "score": round(float(score), 4),
                        "text": s["text"], "page": s["page"],
                        "section": s["section"]})
        return out

    # ---- persistence -------------------------------------------------
    def save(self, directory: str, job_id: str):
        import faiss
        faiss.write_index(self.index, os.path.join(directory, f"{job_id}.faiss"))
        with open(os.path.join(directory, f"{job_id}_sentences.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.sentences, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str, job_id: str) -> "SemanticIndex":
        import faiss
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(os.path.join(directory, f"{job_id}.faiss"))
        with open(os.path.join(directory, f"{job_id}_sentences.json"),
                  encoding="utf-8") as f:
            obj.sentences = json.load(f)
        return obj
