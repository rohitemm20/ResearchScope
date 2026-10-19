"""
relevance.py
------------
Stage: Research-paper relevance scoring.

Compares the research paper against the user's selected:
    - domain
    - subdomains

using the same SentenceTransformer embeddings used by Phase 2.

For each interest term we compare it against:
    1. the most relevant paper sentences
    2. the final paper summary
    3. extracted paper keywords

The result contains:
    - overall relevance score
    - domain score
    - individual subdomain scores
    - matching evidence
    - short justification
"""

import json
import os

import numpy as np

from .pipeline import get_embedder
from .search import SemanticIndex


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

TOP_K_SENTENCES = 5

# Contribution of the three semantic views.
WEIGHT_SENTENCES = 0.60
WEIGHT_SUMMARY = 0.30
WEIGHT_KEYWORDS = 0.10

# Domain/subdomain contribution to overall score.
WEIGHT_DOMAIN = 0.40
WEIGHT_SUBDOMAINS = 0.60


# These are temporary calibration values.
# We will NOT treat them as final until we run the calibration tests.
LO = 0.10
HI = 0.55


# ----------------------------------------------------------------------
# Score conversion
# ----------------------------------------------------------------------

def calibrate_score(raw_similarity: float) -> int:
    """
    Convert cosine similarity into a 0-100 relevance score.

    LO and HI are temporary calibration boundaries.
    We will tune them using real ResearchScope papers.
    """
    score = (raw_similarity - LO) / (HI - LO)
    score = np.clip(score, 0.0, 1.0)
    return int(round(float(score) * 100))


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------

def relevance_label(score: int) -> str:
    if score >= 70:
        return "Strong match"
    if score >= 45:
        return "Partial match"
    if score >= 25:
        return "Weak match"
    return "Not relevant"


# ----------------------------------------------------------------------
# User-interest cleaning
# ----------------------------------------------------------------------

def clean_interests(domain: str, subdomains: list[str]):
    domain = (domain or "").strip()[:100]

    cleaned = []
    seen = set()

    for subdomain in subdomains or []:
        subdomain = subdomain.strip()[:80]

        if not subdomain:
            continue

        key = subdomain.lower()

        if key == domain.lower():
            continue

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(subdomain)

    return domain, cleaned[:8]


# ----------------------------------------------------------------------
# Individual term scoring
# ----------------------------------------------------------------------

def score_term(
    term: str,
    embedder,
    sentence_vectors: np.ndarray,
    sentences: list[dict],
    summary: str,
    keywords: list[str],
):
    """
    Calculate semantic relevance of one user-interest term.
    """

    # --------------------------------------------------------------
    # Encode the user's interest.
    # --------------------------------------------------------------

    query_vector = embedder.encode([term])[0]

    # --------------------------------------------------------------
    # Sentence similarity
    # --------------------------------------------------------------

    sentence_scores = sentence_vectors @ query_vector

    k = min(TOP_K_SENTENCES, len(sentence_scores))

    top_indices = np.argsort(sentence_scores)[::-1][:k]

    top_scores = sentence_scores[top_indices]

    sentence_similarity = float(np.mean(top_scores))

    best_index = int(top_indices[0])

    best_sentence = sentences[best_index]

    # --------------------------------------------------------------
    # Summary similarity
    # --------------------------------------------------------------

    summary_vector = embedder.encode([summary])[0]

    summary_similarity = float(
        query_vector @ summary_vector
    )

    # --------------------------------------------------------------
    # Keyword similarity
    # --------------------------------------------------------------

    keyword_similarity = 0.0

    if keywords:
        keyword_vectors = embedder.encode(keywords)

        keyword_scores = keyword_vectors @ query_vector

        keyword_similarity = float(
            np.max(keyword_scores)
        )

    # --------------------------------------------------------------
    # Combined raw similarity
    # --------------------------------------------------------------

    raw_similarity = (
        WEIGHT_SENTENCES * sentence_similarity
        + WEIGHT_SUMMARY * summary_similarity
        + WEIGHT_KEYWORDS * keyword_similarity
    )

    score = calibrate_score(raw_similarity)

    return {
        "term": term,
        "score": score,
        "raw_similarity": round(raw_similarity, 4),
        "components": {
            "sentences": round(sentence_similarity, 4),
            "summary": round(summary_similarity, 4),
            "keywords": round(keyword_similarity, 4),
        },
        "evidence": {
            "text": best_sentence["text"],
            "page": best_sentence["page"],
            "section": best_sentence["section"],
        },
    }


# ----------------------------------------------------------------------
# Main relevance evaluation
# ----------------------------------------------------------------------

def evaluate_relevance(
    domain: str,
    subdomains: list[str],
    embedder,
    sentence_vectors: np.ndarray,
    sentences: list[dict],
    summary: str,
    keywords: list[str],
):
    domain_result = score_term(
        domain,
        embedder,
        sentence_vectors,
        sentences,
        summary,
        keywords,
    )

    subdomain_results = []

    for subdomain in subdomains:
        result = score_term(
            subdomain,
            embedder,
            sentence_vectors,
            sentences,
            summary,
            keywords,
        )

        subdomain_results.append(result)

    # --------------------------------------------------------------
    # Overall score
    # --------------------------------------------------------------

    if subdomain_results:
        subdomain_average = float(
            np.mean(
                [item["score"] for item in subdomain_results]
            )
        )

        overall_score = int(
            round(
                WEIGHT_DOMAIN * domain_result["score"]
                + WEIGHT_SUBDOMAINS * subdomain_average
            )
        )

    else:
        overall_score = domain_result["score"]

    label = relevance_label(overall_score)

    # --------------------------------------------------------------
    # Justification
    # --------------------------------------------------------------

    all_results = [
        domain_result,
        *subdomain_results,
    ]

    strongest = max(
        all_results,
        key=lambda item: item["score"]
    )

    weakest = min(
        all_results,
        key=lambda item: item["score"]
    )

    justification = build_justification(
        domain,
        overall_score,
        label,
        strongest,
        weakest,
        len(all_results),
    )

    return {
        "score": overall_score,
        "label": label,
        "justification": justification,
        "domain": domain_result,
        "subdomains": subdomain_results,
    }


# ----------------------------------------------------------------------
# Justification
# ----------------------------------------------------------------------

def build_justification(
    domain,
    overall_score,
    label,
    strongest,
    weakest,
    number_of_terms,
):
    if label == "Strong match":
        opening = (
            f"This paper is highly relevant to "
            f"{domain} ({overall_score}/100)."
        )

    elif label == "Partial match":
        opening = (
            f"This paper is partially relevant to "
            f"{domain} ({overall_score}/100)."
        )

    elif label == "Weak match":
        opening = (
            f"This paper has limited relevance to "
            f"{domain} ({overall_score}/100)."
        )

    else:
        opening = (
            f"This paper has low semantic relevance to "
            f"{domain} ({overall_score}/100)."
        )

    evidence = (
        f"The strongest semantic match is "
        f"“{strongest['term']}” with a score of "
        f"{strongest['score']}/100. "
        f"The closest supporting passage is on page "
        f"{strongest['evidence']['page']}, "
        f"in the {strongest['evidence']['section']} section."
    )

    if (
        number_of_terms > 1
        and weakest["score"] < 40
        and weakest["term"] != strongest["term"]
    ):
        weakness = (
            f"However, the paper has weaker semantic alignment "
            f"with “{weakest['term']}” ({weakest['score']}/100)."
        )
    else:
        weakness = ""

    return " ".join(
        part for part in [
            opening,
            evidence,
            weakness,
        ]
        if part
    )


# ----------------------------------------------------------------------
# Load processed paper and calculate relevance
# ----------------------------------------------------------------------

def check_relevance(
    job_id: str,
    out_dir: str,
    domain: str,
    subdomains: list[str],
):
    """
    Load an existing Phase 2 result and calculate relevance.
    """

    phase2_path = os.path.join(
        out_dir,
        f"{job_id}_phase2.json",
    )

    if not os.path.exists(phase2_path):
        raise FileNotFoundError(
            "Paper has not been processed by Phase 2 yet."
        )

    with open(
        phase2_path,
        encoding="utf-8",
    ) as f:
        phase2 = json.load(f)

    # Load the existing sentence embeddings/index.
    index = SemanticIndex.load(
        out_dir,
        job_id,
    )

    sentence_vectors = (
        index.index
        .reconstruct_n(
            0,
            index.index.ntotal,
        )
        .astype("float32")
    )

    sentences = index.sentences

    # Use both keyword sets generated by your existing pipeline.
    keywords = [
        item["keyword"]
        for item in (
            phase2.get("keywords", [])
            + phase2.get("keywords_document", [])
        )
    ]

    # Remove duplicate keywords while preserving order.
    keywords = list(
        dict.fromkeys(keywords)
    )

    return evaluate_relevance(
        domain=domain,
        subdomains=subdomains,
        embedder=get_embedder(),
        sentence_vectors=sentence_vectors,
        sentences=sentences,
        summary=phase2.get("summary", ""),
        keywords=keywords,
    )