
"""
Evidence-based research gap identification.

Detects author-stated limitations, future-work statements, and
potential evaluation or experiment gaps from extracted paper text.

This is a heuristic baseline, not a substitute for expert review.
"""

import re
from collections import Counter


PATTERNS = {
    "limitations": [
        r"\b(?:limitation|limitations|limited by|restricted to)\b",
        r"\b(?:constraint|constraints|drawback|drawbacks)\b",
        r"\b(?:we did not|was not possible|could not)\b",
        r"\b(?:small sample size|small dataset|limited dataset)\b",
        r"\b(?:only considers|only considered|focuses exclusively on)\b",
    ],
    "missing_experiments": [
        r"\b(?:has not been tested|have not been tested)\b",
        r"\b(?:has not been evaluated|have not been evaluated)\b",
        r"\b(?:not evaluated on|not tested on|not validated on)\b",
        r"\b(?:further experiments|additional experiments)\b",
        r"\b(?:requires further validation|needs further validation)\b",
    ],
    "weak_evaluation": [
        r"\b(?:single dataset|one dataset|only one dataset)\b",
        r"\b(?:no baseline|without a baseline|no comparison)\b",
        r"\b(?:not compared against|not compared with)\b",
        r"\b(?:lack of evaluation|limited evaluation)\b",
        r"\b(?:small sample size|limited test set)\b",
        r"\b(?:only one metric|single evaluation metric)\b",
    ],
    "future_opportunities": [
        r"\b(?:future work|future research|future studies)\b",
        r"\b(?:further research|further investigation)\b",
        r"\b(?:remains to be explored|remain unexplored)\b",
        r"\b(?:we plan to|will be investigated)\b",
        r"\b(?:an interesting direction|promising direction)\b",
    ],
}

SUGGESTIONS = {
    "limitations": (
        "Assess whether the stated limitation affects the paper's "
        "claims and investigate ways to address it."
    ),
    "missing_experiments": (
        "Consider additional experiments relevant to the research "
        "question and validate the method under those conditions."
    ),
    "weak_evaluation": (
        "Review whether the evaluation supports the paper's claims; "
        "where appropriate, use suitable baselines, datasets, or metrics."
    ),
    "future_opportunities": (
        "Review the authors' proposed direction and determine a "
        "specific, testable follow-up research question."
    ),
}


def _sentences(text: str) -> list[str]:
    """Split extracted text into candidate sentences."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    parts = re.split(r"(?<=[.!?])\s+|\n", text)

    return [
        re.sub(r"\s+", " ", part).strip()
        for part in parts
        if len(part.strip()) >= 35
    ]


def _matches(sentence: str, patterns: list[str]) -> bool:
    return any(
        re.search(pattern, sentence, re.IGNORECASE)
        for pattern in patterns
    )


def analyze_research_gaps(text: str, job_id: str) -> dict:
    """
    Analyze extracted text without changing the PDF extraction pipeline.

    Explicit statements and heuristic indicators are kept distinct.
    The system does not claim that an unmentioned experiment was
    definitely omitted from the original research.
    """
    if not text or not text.strip():
        raise ValueError("No extracted paper text is available.")

    findings = []
    seen = set()

    for sentence in _sentences(text):
        matched_categories = [
            category
            for category, patterns in PATTERNS.items()
            if _matches(sentence, patterns)
        ]

        for category in matched_categories:
            normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
            key = (category, normalized)

            if key in seen:
                continue

            seen.add(key)

            findings.append({
                "id": f"gap-{len(findings) + 1}",
                "category": category,
                "title": {
                    "limitations": "Potential or stated limitation",
                    "missing_experiments": "Experiment or validation concern",
                    "weak_evaluation": "Evaluation concern",
                    "future_opportunities": "Future research direction",
                }[category],
                "description": sentence,
                "evidence": sentence,
                "evidence_type": "author_stated",
                "interpretation": (
                    "The paper explicitly discusses this limitation, "
                    "evaluation concern, or future direction. Review the "
                    "sentence in context before treating it as a research gap."
                ),
                "suggested_investigation": SUGGESTIONS[category],
                "confidence": "candidate",
            })

    counts = Counter(item["category"] for item in findings)

    return {
        "job_id": job_id,
        "method": "rule_based_evidence_detection",
        "notice": (
            "These are candidate findings, not verified research gaps. "
            "Keyword patterns may miss relevant passages or flag statements "
            "that are not genuine weaknesses. No experiment is assumed "
            "missing merely because it is not mentioned."
        ),
        "total_candidates": len(findings),
        "counts": {
            category: counts.get(category, 0)
            for category in PATTERNS
        },
        "research_gaps": findings,
    }
