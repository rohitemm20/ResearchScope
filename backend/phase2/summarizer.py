
"""
summarizer.py
-------------
Non-generative, accuracy-first scientific summarization.

Uses sentence selection rather than a generative language model.
It does not load BART, DistilBART, Transformers, or a text-generation model.

Main strategy:
1. Rank sentences using existing sentence embeddings.
2. Prefer scientific sections such as Abstract, Methods, Results,
   Discussion, Limitations, and Conclusion.
3. Reduce repeated information.
4. Preserve source sentence wording and original order.
5. Produce one consolidated paper-level summary.
"""

import re
from collections import Counter

import numpy as np


# Kept for compatibility with the existing pipeline and API's
# quality=fast|best parameter. These are modes, not model names.
MODELS = {
    "fast": "extractive_fast",
    "best": "extractive_balanced",
}

SECTION_WEIGHTS = {
    "abstract": 1.45,
    "introduction": 1.05,
    "background": 0.90,
    "related work": 0.80,
    "literature review": 0.80,
    "method": 1.00,
    "methodology": 1.00,
    "materials and methods": 1.00,
    "experiment": 1.10,
    "experiments": 1.10,
    "result": 1.40,
    "results": 1.40,
    "finding": 1.35,
    "findings": 1.35,
    "discussion": 1.25,
    "limitation": 1.25,
    "limitations": 1.25,
    "conclusion": 1.35,
    "conclusions": 1.35,
    "future work": 1.00,
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _section_weight(section: str) -> float:
    """Return a conservative weight based on the section heading."""
    name = _normalise(section).lower()

    # Match longer heading labels first.
    for key in sorted(SECTION_WEIGHTS, key=len, reverse=True):
        if key in name:
            return SECTION_WEIGHTS[key]

    return 0.85


def _sentence_key(text: str) -> str:
    """Normalised key for exact/near-exact textual duplicates."""
    text = text.lower()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = re.sub(r"[^a-z0-9<> ]+", " ", text)
    return _normalise(text)


def _is_usable_sentence(text: str) -> bool:
    """Reject obvious fragments and likely extraction artefacts."""
    text = _normalise(text)

    if len(text.split()) < 7:
        return False

    if len(text) > 1200:
        return False

    if not re.search(r"[.!?][\"')\]]?\s*$", text):
        return False

    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(len(text), 1) < 0.55:
        return False

    return True


class Summarizer:
    """
    Drop-in replacement for the former transformer-backed Summarizer.

    model_name remains accepted for compatibility with pipeline.py,
    but no model is downloaded or loaded.
    """

    def __init__(
        self,
        model_name: str = MODELS["fast"],
        max_input_tokens: int = 900,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count without a model tokenizer.
        Used only for chunk-size budgeting.
        """
        return len(re.findall(r"\w+|[^\w\s]", text or ""))

    def summarize_text(self, text: str, max_len: int = 160) -> str:
        """
        Compatibility method: select representative source sentences.

        This method does not paraphrase or generate new sentences.
        For best results, use summarize_sentences() with embeddings.
        """
        text = _normalise(text)
        if not text:
            return ""

        from .nlp_utils import split_sentences

        paragraphs = [{
            "page": 1,
            "section": "Document",
            "text": text,
            "block_ids": [],
            "is_heading": False,
        }]
        sentences = split_sentences(paragraphs)

        if not sentences:
            return text

        words_budget = max(40, int(max_len * 1.5))
        chosen = []
        used = 0

        # Earlier sentences are a safe fallback when embeddings
        # are not supplied. Keep source order and original wording.
        for sentence in sentences:
            candidate = sentence["text"]
            n_words = len(candidate.split())

            if chosen and used + n_words > words_budget:
                continue

            chosen.append(candidate)
            used += n_words

            if used >= words_budget:
                break

        return " ".join(chosen) if chosen else sentences[0]["text"]

    def summarize_sentences(
        self,
        sentences: list[dict],
        embeddings: np.ndarray,
        max_sentences: int = 12,
        max_words: int = 350,
    ) -> str:
        """
        Build a single paper-level summary from source sentences.

        Sentences are ranked for importance, selected under a word budget,
        and then returned in their original document order.
        """
        if not sentences:
            return ""

        vectors = np.asarray(embeddings, dtype=np.float32)

        if len(vectors) != len(sentences):
            raise ValueError(
                "Sentence/embedding count mismatch in summarizer."
            )

        if len(sentences) == 1:
            return sentences[0]["text"]

        valid = [
            i for i, sentence in enumerate(sentences)
            if _is_usable_sentence(sentence.get("text", ""))
        ]

        if not valid:
            # Avoid returning an empty summary for unusual PDFs.
            valid = list(range(len(sentences)))

        matrix = vectors[valid]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-9)

        # Centrality: similarity to the average semantic content.
        centroid = matrix.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        centrality = matrix @ centroid

        # Small positional preference: early sentences can establish
        # the paper's objective, but later results remain competitive.
        scores = {}
        for local_i, original_i in enumerate(valid):
            sentence = sentences[original_i]
            text = _normalise(sentence.get("text", ""))
            section = sentence.get("section", "")

            score = float(centrality[local_i])
            score *= _section_weight(section)

            # Abstract and conclusion are useful, but avoid allowing
            # the abstract alone to crowd out all results/methods.
            position_bonus = 1.0 / (1.0 + original_i / 80.0)
            score *= 0.92 + 0.08 * position_bonus

            # Prefer information-bearing sentences without rewarding
            # very long sentences excessively.
            word_count = max(len(text.split()), 1)
            score *= min(1.0, 22.0 / word_count) if word_count > 22 else 1.0

            # Sentences containing quantitative evidence can be useful.
            if re.search(r"\b\d+(?:\.\d+)?\s*%|\bp\s*[=<]\s*0?\.\d+", text, re.I):
                score *= 1.06

            scores[original_i] = score

        ranked = sorted(scores, key=scores.get, reverse=True)

        selected = []
        selected_keys = set()
        selected_vectors = []
        used_words = 0

        for idx in ranked:
            text = _normalise(sentences[idx].get("text", ""))
            key = _sentence_key(text)
            words = len(text.split())

            if not text or key in selected_keys:
                continue

            if used_words + words > max_words:
                continue

            # Semantic redundancy guard. The pipeline already has a
            # redundancy filter; this is a second, summary-only guard.
            vec = vectors[idx]
            vec = vec / max(float(np.linalg.norm(vec)), 1e-9)

            if selected_vectors:
                similarities = [
                    float(np.dot(vec, other))
                    for other in selected_vectors
                ]
                if max(similarities) >= 0.88:
                    continue

            selected.append(idx)
            selected_keys.add(key)
            selected_vectors.append(vec)
            used_words += words

            if len(selected) >= max_sentences or used_words >= max_words:
                break

        # If the word budget was too strict, retain the best sentence.
        if not selected:
            selected = [ranked[0]]

        # Restore the paper's order so results/methods aren't shuffled.
        selected.sort()
        return " ".join(_normalise(sentences[i]["text"]) for i in selected)

    def summarize_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Compatibility method for existing callers.
        Chunk summaries are source-sentence selections, not generations.
        """
        for chunk in chunks:
            chunk["summary"] = self.summarize_text(
                chunk.get("text", ""),
                max_len=150,
            )
        return chunks

    def final_summary(
        self,
        chunk_summaries: list[str],
        final_max_len: int = 220,
        _depth: int = 0,
    ) -> str:
        """
        Compatibility fallback for older callers.
        The updated pipeline should use summarize_sentences() instead.
        """
        text = " ".join(s for s in chunk_summaries if s)
        return self.summarize_text(text, max_len=final_max_len)
