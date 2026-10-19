"""
keywords.py
-----------
Stage: Final Summary -> Keyword Extraction (KeyBERT)

Re-uses the already loaded SentenceTransformer so no second model is loaded.
MMR keeps keywords diverse instead of 10 variants of "pricing".
"""


def extract_keywords(text: str, st_model, top_n: int = 10,
                     ngram_range=(1, 3), diversity: float = 0.5) -> list[dict]:
    from keybert import KeyBERT
    kb = KeyBERT(model=st_model)
    pairs = kb.extract_keywords(
        text, keyphrase_ngram_range=ngram_range, stop_words="english",
        use_mmr=True, diversity=diversity, top_n=top_n,
    )
    return [{"keyword": k, "score": round(float(s), 4)} for k, s in pairs]
