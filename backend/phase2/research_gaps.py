"""
Evidence-based research gap identification (v2).

Finds author-stated limitations, evaluation concerns, missing validation and
future-work statements in a paper, and ranks them by how trustworthy the
signal is.

What changed compared with v1
-----------------------------
* Works on cleaned, sectioned paragraphs (references, captions, table cells
  and running headers are already removed) instead of the raw .txt dump, so
  findings carry page + section provenance and bibliography noise is gone.
* Uses the shared abbreviation-aware sentence splitter ("et al.", "Fig.").
* Each finding has a *scope*: ``this_work`` (the authors talk about their own
  study) or ``prior_work`` (the authors criticise existing work, which is the
  motivation for the paper rather than a weakness of it).
* Each finding has a numeric ``score`` and a ``confidence`` of high / medium /
  low, based on section, phrase strength, scope, and hedging. Results are
  sorted best-first.
* Weak, over-broad triggers (a bare "could not", "constraint") were replaced
  with stronger phrase patterns; the matched phrase is returned so the UI can
  highlight it.
* The same sentence is not reported under several near-identical categories
  unless it truly matches strongly in each.
* Neighbouring sentences are returned as context.

This is still a heuristic baseline, not a substitute for expert review.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

METHOD_NAME = "rule_based_evidence_detection_v2"

# --------------------------------------------------------------------------
# Patterns. Each entry is (regex, weight). Weight reflects how reliably the
# phrase signals a genuine statement of that kind (0..1).
# --------------------------------------------------------------------------
_P = re.compile
PATTERNS: dict[str, list[tuple[re.Pattern, float]]] = {
    "limitations": [
        (_P(r"\blimitations? (?:of|in|to|is|are|include|includes)\b", re.I), 0.9),
        (_P(r"\b(?:is|are|remains?|be) limited (?:by|to|in)\b", re.I), 0.8),
        (_P(r"\b(?:we|this (?:study|work|paper)) (?:did|do|does) not (?:consider|account for|address|investigate|include|explore|cover|evaluate|test)\b", re.I), 0.85),
        (_P(r"\b(?:does|do|did) not (?:generali[sz]e|scale|transfer)\b", re.I), 0.75),
        (_P(r"\b(?:beyond the scope|out of scope|outside the scope)\b", re.I), 0.8),
        (_P(r"\b(?:small|limited|insufficient) (?:sample size|dataset|data|number of (?:samples|participants|subjects))\b", re.I), 0.8),
        (_P(r"\b(?:only considers?|only considered|only focus(?:es|ed) on|focus(?:es|ed) exclusively on|restricted to)\b", re.I), 0.7),
        (_P(r"\b(?:threats? to validity|potential bias|selection bias|confounding)\b", re.I), 0.75),
        (_P(r"\b(?:drawbacks?|shortcomings?|caveats?)\b", re.I), 0.65),
        (_P(r"\bwe were unable to\b|\bwe could not\b", re.I), 0.6),
    ],
    "missing_experiments": [
        (_P(r"\b(?:has|have|had) (?:not|yet to) been (?:tested|evaluated|validated|verified|explored|studied|examined)\b", re.I), 0.9),
        (_P(r"\b(?:not|never) (?:been )?(?:evaluated|tested|validated|verified) (?:on|in|with|against|under)\b", re.I), 0.9),
        (_P(r"\b(?:further|additional|more) (?:experiments?|experimentation|evaluation|validation|testing|ablations?)\b", re.I), 0.85),
        (_P(r"\b(?:needs?|requires?|warrants?|calls? for|merits?) (?:further|more|additional) (?:validation|testing|evaluation|study|investigation|experiments?)\b", re.I), 0.9),
        (_P(r"\b(?:real-world|large-scale|field|clinical|longitudinal|cross-domain) (?:validation|evaluation|testing|deployment|trials?|studies)\b[^.]{0,60}\b(?:needed|required|remain|future|not)\b", re.I), 0.75),
        (_P(r"\b(?:untested|unvalidated|unverified)\b", re.I), 0.7),
    ],
    "weak_evaluation": [
        (_P(r"\b(?:only|just|solely) (?:one|a single|1) (?:dataset|benchmark|corpus|domain|task|metric|baseline|model)\b", re.I), 0.9),
        (_P(r"\b(?:single|one) (?:dataset|benchmark|corpus|evaluation metric|test set)\b", re.I), 0.7),
        (_P(r"\b(?:no|without|lack(?:s|ing)? (?:of )?(?:a |any )?)(?:baselines?|comparison|ablation|statistical (?:test|significance)|confidence intervals?|error bars?)\b", re.I), 0.8),
        (_P(r"\bnot (?:compared|benchmarked|contrasted) (?:against|with|to)\b", re.I), 0.9),
        (_P(r"\b(?:lack|absence) of (?:rigorous |extensive |thorough |systematic )?(?:evaluation|validation|benchmarking)\b", re.I), 0.85),
        (_P(r"\blimited (?:evaluation|test set|benchmark|validation)\b", re.I), 0.85),
        (_P(r"\bnot statistically significant\b|\bwithout statistical\b", re.I), 0.75),
        (_P(r"\b(?:evaluated|tested|validated) (?:only|solely) (?:on|in|with)\b", re.I), 0.9),
    ],
    "future_opportunities": [
        (_P(r"\bfuture (?:work|research|studies|study|directions?|investigations?|efforts?)\b", re.I), 0.9),
        (_P(r"\b(?:further|more) (?:research|investigation|study|work|exploration)\b", re.I), 0.7),
        (_P(r"\b(?:remains?|remain) (?:an )?(?:open|unexplored|unresolved|to be (?:explored|investigated|studied|addressed))\b", re.I), 0.9),
        (_P(r"\bwe (?:plan|intend|aim|hope|will|would like) to (?:extend|explore|investigate|study|apply|evaluate|address|improve|incorporate)\b", re.I), 0.85),
        (_P(r"\b(?:will|should|could|can) be (?:investigated|explored|extended|addressed|studied) in (?:future|subsequent|later)\b", re.I), 0.9),
        (_P(r"\b(?:promising|interesting|natural|important) (?:direction|avenue|extension|path|opportunity)s?\b", re.I), 0.85),
        (_P(r"\bopen (?:problems?|questions?|challenges?|research (?:questions?|problems?))\b", re.I), 0.7),
    ],
}

TITLES = {
    "limitations": "Stated limitation",
    "missing_experiments": "Missing experiment or validation",
    "weak_evaluation": "Evaluation concern",
    "future_opportunities": "Future research direction",
}

SUGGESTIONS = {
    "limitations": (
        "Assess whether this limitation undermines the paper's claims, then "
        "design a study or method change that removes or measures it."
    ),
    "missing_experiments": (
        "Run the untested condition (new dataset, domain, scale or setting) "
        "and report whether the original result still holds."
    ),
    "weak_evaluation": (
        "Re-evaluate with stronger baselines, additional datasets or metrics, "
        "and statistical testing, then compare against the reported numbers."
    ),
    "future_opportunities": (
        "Turn the authors' proposed direction into one specific, testable "
        "research question with a clear success criterion."
    ),
}

# --------------------------------------------------------------------------
# Context signals
# --------------------------------------------------------------------------
# Sections where authors normally reflect on their own work.
_REFLECTIVE_SECTION = re.compile(
    r"limitation|discussion|conclusion|future|threats|outlook|concluding|"
    r"summary|implications|challenges|open (?:problems|questions)", re.I)
# Sections where critique usually targets other people's work.
_BACKGROUND_SECTION = re.compile(
    r"related work|background|literature|prior work|previous work|"
    r"state of the art|introduction|motivation|overview", re.I)

# Sentence is about the authors' own work.
_SELF_REF = re.compile(
    r"\b(?:we|our|ours|us|this (?:paper|study|work|article|approach|method|"
    r"framework|system|model|analysis)|the (?:proposed|present|current) "
    r"(?:approach|method|framework|system|model|study|work))\b", re.I)
# Sentence is about other people's work.
_OTHER_REF = re.compile(
    r"\b(?:existing|previous|prior|earlier|state-of-the-art|conventional|"
    r"traditional|current (?:methods|approaches|systems|techniques)|"
    r"most (?:methods|approaches|studies|works)|"
    r"many (?:methods|approaches|studies|works)|"
    r"several (?:methods|approaches|studies|works)|"
    r"others?|researchers|et al\.?)\b|\[\d+(?:[,\u2013\-]\s*\d+)*\]", re.I)

# Hedging words slightly raise trust that the authors mean a real caveat.
_HEDGE = re.compile(
    r"\b(?:however|nevertheless|although|despite|unfortunately|admittedly|nonetheless)\b",
    re.I)

# Boilerplate such as "no limitations" or "overcomes the drawbacks of ...".
_NEGATED = re.compile(
    r"\b(?:no|without any|free of|not subject to|overcomes?|addresses|alleviates|"
    r"removes?|eliminates?|mitigates?) (?:major |significant |notable |such )?"
    r"(?:limitations?|drawbacks?|shortcomings?)\b",
    re.I)

MIN_SENTENCE_CHARS = 35
MAX_SENTENCE_CHARS = 700
MAX_FINDINGS = 60

# Fallback splitter when spaCy is unavailable. Protects common abbreviations.
_ABBR = re.compile(r"\b(?:et al|e\.g|i\.e|Figs?|Eqs?|vs|etc|approx|No|Dr|Prof)\.$", re.I)


def _fallback_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)
    out: list[str] = []
    buf = ""
    for piece in pieces:
        buf = f"{buf} {piece}".strip()
        if _ABBR.search(buf):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _units_from_parsed(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Sentences with page/section provenance from the parsed PDF JSON."""
    from .nlp_utils import split_sentences
    from .text_cleaning import clean_document

    rows = split_sentences(clean_document(parsed), min_words=5)
    return [
        {"text": _normalise(r["text"]), "page": r.get("page"),
         "section": r.get("section"), "para_idx": r.get("para_idx")}
        for r in rows
    ]


def _units_from_text(text: str) -> list[dict[str, Any]]:
    """Plain-text fallback: no page or section information."""
    text = re.sub(r"[ \t]+", " ", text)
    units: list[dict[str, Any]] = []
    for para_idx, para in enumerate(re.split(r"\n\s*\n", text)):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if not para:
            continue
        for sent in _fallback_split(para):
            sent = _normalise(sent)
            if len(sent.split()) >= 5:
                units.append({"text": sent, "page": None, "section": None,
                              "para_idx": para_idx})
    return units


def _scope(sentence: str, section: str | None) -> str:
    """Decide whether the sentence criticises this paper or other work."""
    self_hits = len(_SELF_REF.findall(sentence))
    other_hits = len(_OTHER_REF.findall(sentence))
    in_background = bool(section and _BACKGROUND_SECTION.search(section)
                         and not _REFLECTIVE_SECTION.search(section))
    if self_hits > other_hits:
        return "this_work"
    if other_hits > self_hits:
        return "prior_work"
    return "prior_work" if in_background and not self_hits else "this_work"


def _score(weight: float, scope: str, section: str | None, sentence: str,
           n_matches: int) -> float:
    score = weight
    if section and _REFLECTIVE_SECTION.search(section):
        score += 0.20
    elif section and _BACKGROUND_SECTION.search(section):
        score -= 0.10
    if scope == "prior_work":
        score -= 0.25
    if _SELF_REF.search(sentence) and scope == "this_work":
        score += 0.05
    if _HEDGE.search(sentence):
        score += 0.05
    if n_matches > 1:
        score += min(0.10, 0.05 * (n_matches - 1))
    if len(sentence) > 450:
        score -= 0.05
    return round(max(0.0, min(1.0, score)), 3)


def _confidence(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.62:
        return "medium"
    return "low"


def _interpretation(category: str, scope: str, confidence: str) -> str:
    if scope == "prior_work":
        return ("The authors describe a weakness in earlier or existing work. "
                "This is usually the motivation for the paper, so it may be an "
                "open problem the paper itself already addresses.")
    base = {
        "limitations": "The authors state a limitation of their own work.",
        "missing_experiments": "The authors indicate something that was not tested or validated.",
        "weak_evaluation": "The text points to a possible weakness in how the work was evaluated.",
        "future_opportunities": "The authors propose a direction they have not yet pursued.",
    }[category]
    if confidence == "low":
        return base + " The match is weak, so read it in context before relying on it."
    return base + " Check the surrounding text to confirm the scope."


def _find_matches(sentence: str) -> dict[str, tuple[float, list[str]]]:
    """category -> (best weight, matched phrases)."""
    found: dict[str, tuple[float, list[str]]] = {}
    for category, rules in PATTERNS.items():
        best = 0.0
        phrases: list[str] = []
        for rx, weight in rules:
            m = rx.search(sentence)
            if m:
                best = max(best, weight)
                phrase = _normalise(m.group(0))
                if phrase.lower() not in (p.lower() for p in phrases):
                    phrases.append(phrase)
        if phrases:
            found[category] = (best, phrases)
    return found


def analyze_research_gaps(
    source: str | dict,
    job_id: str,
    max_findings: int = MAX_FINDINGS,
) -> dict:
    """
    ``source`` is either the parsed-PDF dict (preferred, gives page and section
    provenance) or the plain extracted text (fallback).
    """
    if isinstance(source, dict):
        units = _units_from_parsed(source)
        input_mode = "parsed_json"
    else:
        if not source or not source.strip():
            raise ValueError("No extracted paper text is available.")
        units = _units_from_text(source)
        input_mode = "plain_text"

    if not units:
        raise ValueError("No usable sentences were found in the extracted text.")

    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for idx, unit in enumerate(units):
        sentence = unit["text"]
        if not (MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS):
            continue
        if _NEGATED.search(sentence):
            continue

        matches = _find_matches(sentence)
        if not matches:
            continue

        # Keep the strongest category; keep a second one only if it is nearly
        # as strong. Prevents one sentence appearing under four headings.
        ranked = sorted(matches.items(), key=lambda kv: -kv[1][0])
        top_weight = ranked[0][1][0]
        chosen = [(c, v) for c, v in ranked if v[0] >= top_weight - 0.1][:2]

        scope = _scope(sentence, unit["section"])
        normalised = re.sub(r"\W+", " ", sentence.lower()).strip()

        for category, (weight, phrases) in chosen:
            key = (category, normalised)
            if key in seen:
                continue
            seen.add(key)

            score = _score(weight, scope, unit["section"], sentence, len(phrases))
            confidence = _confidence(score)

            before = units[idx - 1]["text"] if idx > 0 else ""
            after = units[idx + 1]["text"] if idx + 1 < len(units) else ""

            findings.append({
                "category": category,
                "title": TITLES[category],
                "description": sentence,
                "evidence": sentence,
                "evidence_type": ("author_stated" if scope == "this_work"
                                  else "prior_work_critique"),
                "scope": scope,
                "matched_phrases": phrases,
                "page": unit["page"],
                "section": unit["section"],
                "context_before": before[:400],
                "context_after": after[:400],
                "interpretation": _interpretation(category, scope, confidence),
                "suggested_investigation": SUGGESTIONS[category],
                "score": score,
                "confidence": confidence,
            })

    findings.sort(key=lambda f: (-f["score"], f["page"] if f["page"] is not None else 10**6))
    total_before_cap = len(findings)
    findings = findings[:max_findings]
    for number, finding in enumerate(findings, start=1):
        finding["id"] = f"gap-{number}"

    counts = Counter(f["category"] for f in findings)
    by_confidence = Counter(f["confidence"] for f in findings)
    by_scope = Counter(f["scope"] for f in findings)

    return {
        "job_id": job_id,
        "method": METHOD_NAME,
        "input_mode": input_mode,
        "notice": (
            "These are candidate findings, not verified research gaps. Keyword "
            "patterns can miss relevant passages or flag statements that are "
            "not genuine weaknesses. Nothing is assumed missing merely because "
            "it is not mentioned."
        ),
        "total_candidates": len(findings),
        "total_before_cap": total_before_cap,
        "sentences_scanned": len(units),
        "counts": {c: counts.get(c, 0) for c in PATTERNS},
        "confidence_counts": {c: by_confidence.get(c, 0) for c in ("high", "medium", "low")},
        "scope_counts": {s: by_scope.get(s, 0) for s in ("this_work", "prior_work")},
        "research_gaps": findings,
    }
