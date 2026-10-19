"""
embeddings.py
-------------
Stage: spaCy Tokenization -> SentenceTransformer Embeddings
"""
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, fast


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """L2-normalised float32 matrix (n, d) -> dot product == cosine."""
        vecs = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.ascontiguousarray(vecs, dtype="float32")
