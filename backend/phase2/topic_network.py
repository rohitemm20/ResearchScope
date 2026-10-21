"""
Research DNA topic network.

Uses the existing Phase 2 sentence embedder and sentence splitter. Topic
importance is estimated from semantic coverage of each source passage, not
raw term counts, query similarity, or model confidence.

Edges are intentionally conservative: an edge is emitted only when an
explicit relation cue is found in a source sentence and both topic labels
occur in that sentence. Semantic similarity alone never creates an edge.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

import numpy as np

from .nlp_utils import split_sentences
from .pipeline import get_embedder
from .text_cleaning import clean_document, load_parsed_json

MAX_TOPICS = 12
MIN_TOPIC_SUPPORT = 2
MIN_ASSIGNMENT_SIMILARITY = 0.22

# These cues are deliberately explicit. Extend only with tested, domain-safe
# patterns. "related_to" is not inferred from embeddings alone.
RELATION_CUES = [
    ("evaluated_by", re.compile(r"\b(?:evaluated|tested|measured|assessed)\s+(?:using|with|by|against)\b", re.I)),
    ("improves", re.compile(r"\b(?:improves|enhances|increases|boosts|reduces|mitigates)\b", re.I)),
    ("uses", re.compile(r"\b(?:uses|using|employs|leverages|adopts|applies)\b", re.I)),
    ("depends_on", re.compile(r"\b(?:depends on|is based on|relies on|requires)\b", re.I)),
    ("compared_with", re.compile(r"\b(?:compared with|compared to|in contrast to|outperforms)\b", re.I)),
    ("produces", re.compile(r"\b(?:produces|generates|predicts|classifies|detects|outputs)\b", re.I)),
]

def _stable_id(prefix: str, value: str) -> str:
    return prefix + "_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]

def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

_LEADING_NOISE = {
    "the", "a", "an", "new", "novel", "proposed", "previously", "recent",
    "various", "several", "different", "following", "using", "based",
    "results", "result", "approach", "method", "methods", "system", "systems",
}
_GENERIC_ONLY = {
    "paper", "study", "work", "result", "results", "approach", "method",
    "methods", "system", "systems", "model", "models", "technique", "techniques",
    "process", "processes", "thing", "things", "value", "values",
}
_ACRONYMS = {
    "ai": "AI", "ml": "ML", "iot": "IoT", "tls": "TLS", "tlsh": "TLSH",
    "api": "API", "apis": "APIs", "cnn": "CNN", "cnns": "CNNs",
    "rnn": "RNN", "rnns": "RNNs", "svm": "SVM", "svms": "SVMs",
    "nlp": "NLP", "llm": "LLM", "llms": "LLMs", "gpu": "GPU", "cpu": "CPU",
    "tcp": "TCP", "ip": "IP", "http": "HTTP", "https": "HTTPS",
    "dns": "DNS", "dos": "DoS", "ddos": "DDoS",
}

def _clean_candidate(phrase: str) -> str | None:
    """Normalize a candidate phrase and reject fragments/generic-only labels."""
    phrase = phrase.replace("’", "'")
    phrase = re.sub(r"\b(?:'s|’s)\b", "", phrase, flags=re.I)
    phrase = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    words = re.findall(r"[A-Za-z][A-Za-z0-9.+_-]*", phrase)
    if not words or len(words) > 5:
        return None

    while words and words[0].lower() in _LEADING_NOISE:
        words.pop(0)
    while len(words) > 1 and words[-1].lower() in {"paper", "study", "work", "approach"}:
        words.pop()
    if not words:
        return None

    lowered = [w.lower() for w in words]
    if len(lowered) == 1 and (lowered[0] in _GENERIC_ONLY or len(lowered[0]) < 3):
        return None
    if all(w in _GENERIC_ONLY for w in lowered):
        return None
    if len(lowered) > 1 and lowered[0] in {"and", "or", "but", "with", "from", "for", "of", "to"}:
        return None

    normalized = " ".join(lowered)
    # Reject sentence-like phrases and common extraction debris.
    if re.search(r"\b(?:is|are|was|were|has|have|had|can|could|would|should|will|may|might)\b", normalized):
        return None
    if len(normalized) < 3:
        return None
    return normalized

def _display_label(term: str) -> str:
    """Human-readable label; keep common technical acronyms recognizable."""
    words = term.split()
    shown = []
    for word in words:
        key = word.lower().strip(".,:;()[]{}")
        shown.append(_ACRONYMS.get(key, word.capitalize()))
    return " ".join(shown)

def _topic_candidates(sentences: list[dict[str, Any]]) -> list[str]:
    """Generate cleaned noun-phrase/entity candidates; fallback only proposes candidates."""
    candidates: dict[str, int] = defaultdict(int)
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        for item in sentences:
            doc = nlp(item["text"][:10000])
            spans = []
            if doc.has_annotation("DEP"):
                spans.extend(doc.noun_chunks)
            spans.extend(ent for ent in doc.ents)
            for span in spans:
                cleaned = _clean_candidate(span.text)
                if cleaned:
                    candidates[cleaned] += 1
    except Exception:
        # The lexical fallback below is used if spaCy or its model is unavailable.
        pass

    if not candidates:
        stop = set("""the a an and or but for with from into over under using use based
        this that these those their its our we they is are was were be been being
        to of in on as by at it can may might will would should have has had""".split())
        for item in sentences:
            words = re.findall(r"\b[A-Za-z][A-Za-z0-9.+_-]{2,}\b", item["text"])
            for word in words:
                if word.lower() not in stop:
                    cleaned = _clean_candidate(word)
                    if cleaned:
                        candidates[cleaned] += 1
            # Only use adjacent noun-like tokens as fallback candidates, and
            # clean them with the same rules to prevent sentence fragments.
            for a, b in zip(words, words[1:]):
                if a.lower() not in stop and b.lower() not in stop:
                    cleaned = _clean_candidate(f"{a} {b}")
                    if cleaned:
                        candidates[cleaned] += 1

    ranked = sorted(candidates, key=lambda term: (-candidates[term], term))
    kept: list[str] = []
    for term in ranked:
        if any(
            term != other
            and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", other)
            and candidates[other] >= candidates[term]
            for other in kept
        ):
            continue
        kept.append(term)
    return kept[:100]

def build_topic_network(parsed: dict[str, Any], max_topics: int = MAX_TOPICS) -> dict[str, Any]:
    paragraphs = clean_document(parsed)
    sentence_rows = split_sentences(paragraphs)
    if not sentence_rows:
        return {"job_id": parsed.get("job_id"), "nodes": [], "edges": [],
                "method": {"name": "semantic_passage_coverage_v1"},
                "message": "No usable extracted text was available."}

    # Keep provenance from the parser's existing page/block data. No heading
    # detection is added here.
    sentences = []
    for row in sentence_rows:
        sentences.append({
            "id": str(row.get("id", len(sentences))),
            "text": row["text"],
            "page": row.get("page"),
            "section": row.get("section"),
            "block_ids": row.get("block_ids", []),
        })

    candidates = _topic_candidates(sentences)
    if not candidates:
        return {"job_id": parsed.get("job_id"), "nodes": [], "edges": [],
                "method": {"name": "semantic_passage_coverage_v1"},
                "message": "No stable topic candidates could be extracted."}

    embedder = get_embedder()
    passage_vecs = np.asarray(embedder.encode([s["text"] for s in sentences]), dtype=float)
    topic_vecs = np.asarray(embedder.encode(candidates), dtype=float)
    # Normalize to cosine similarity regardless of model output convention.
    passage_vecs /= np.maximum(np.linalg.norm(passage_vecs, axis=1, keepdims=True), 1e-12)
    topic_vecs /= np.maximum(np.linalg.norm(topic_vecs, axis=1, keepdims=True), 1e-12)
    similarities = passage_vecs @ topic_vecs.T

    support = defaultdict(list)
    raw_contribution = defaultdict(float)
    for i, sent in enumerate(sentences):
        order = np.argsort(similarities[i])[::-1]
        assigned = []
        for j in order:
            sim = float(similarities[i, j])
            if sim < MIN_ASSIGNMENT_SIMILARITY:
                break
            term = candidates[int(j)]
            # A passage can contribute to at most two distinct topics. This
            # prevents long papers from assigning every sentence to every node.
            if term not in assigned:
                assigned.append(term)
            if len(assigned) >= 2:
                break
        if not assigned:
            continue
        # Passage mass sums to 1. Topic mass is based on semantic coverage;
        # each passage's contribution is divided among its assigned topics.
        weight = 1.0 / len(assigned)
        for term in assigned:
            support[term].append((i, float(similarities[i, candidates.index(term)])))
            raw_contribution[term] += weight

    viable = [t for t in candidates if len(support[t]) >= MIN_TOPIC_SUPPORT]
    viable = sorted(viable, key=lambda t: (-raw_contribution[t], t))[:max_topics]
    if not viable and sentences:
        # For very short papers, permit single-passage support rather than
        # fabricating topics; the evidence is still attached to the passage.
        viable = [t for t in candidates if support[t]]
        viable = sorted(viable, key=lambda t: (-raw_contribution[t], t))[:max_topics]

    total = sum(raw_contribution[t] for t in viable)
    if total <= 0:
        return {"job_id": parsed.get("job_id"), "nodes": [], "edges": [],
                "method": {"name": "semantic_passage_coverage_v1"},
                "message": "Text was available but no topics met the support threshold."}

    # Deterministic two-decimal percentages with largest-remainder correction
    # ensure displayed values sum to exactly 100.00.
    exact = {t: raw_contribution[t] / total * 100 for t in viable}
    rounded = {t: round(exact[t], 2) for t in viable}
    delta = round(100.0 - sum(rounded.values()), 2)
    if viable:
        rounded[max(viable, key=lambda t: (exact[t] - rounded[t], raw_contribution[t], t))] += delta

    nodes = []
    topic_passages = {}
    for term in viable:
        rows = sorted(support[term], key=lambda x: (-x[1], x[0]))
        evidence = []
        for i, sim in rows[:5]:
            evidence.append({
                "text": sentences[i]["text"],
                "page": sentences[i]["page"],
                "section": sentences[i]["section"],
                "block_ids": sentences[i]["block_ids"],
                "similarity": round(sim, 4),
                "status": "source_passage",
            })
        topic_passages[term] = [sentences[i] for i, _ in rows]
        nodes.append({
            "id": _stable_id("topic", term),
            "label": _display_label(term),
            "percentage": rounded[term],
            "passage_support": len(rows),
            "description": f"Supported by {len(rows)} semantically assigned source passage(s).",
            "evidence": evidence,
        })

    # Evidence-grounded edges only. Both labels must literally occur in the
    # same source sentence and a typed relation cue must be present. Similarity
    # is retained only as a separately named strength diagnostic.
    edges_by_key = {}
    norm_to_term = {_normal(t): t for t in viable}
    for sent in sentences:
        low = _normal(sent["text"])
        present = [t for t in viable if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", low)]
        if len(present) < 2:
            continue
        cue = next(((name, rx) for name, rx in RELATION_CUES if rx.search(sent["text"])), None)
        if not cue:
            continue
        relation_type, cue_rx = cue
        for ix, left in enumerate(present):
            for right in present[ix + 1:]:
                a, b = sorted((left, right))
                key = (a, b, relation_type)
                rec = edges_by_key.get(key)
                if rec is None:
                    rec = {"a": a, "b": b, "type": relation_type, "evidence": []}
                    edges_by_key[key] = rec
                rec["evidence"].append({
                    "text": sent["text"], "page": sent["page"],
                    "section": sent["section"], "block_ids": sent["block_ids"],
                    "status": "validated_relation_cue",
                    "cue": cue_rx.search(sent["text"]).group(0),
                })

    node_by_term = {term: node for term, node in zip(viable, nodes)}
    edges = []
    for (a, b, rel), rec in sorted(edges_by_key.items()):
        # Edge strength is an independently calculated structural diagnostic:
        # number of unique evidence sentences, scaled to [0,1]. It is NOT a
        # topic percentage and is not described as probability/confidence.
        unique_evidence = {e["text"] for e in rec["evidence"]}
        strength = min(1.0, len(unique_evidence) / 3.0)
        edges.append({
            "id": _stable_id("edge", f"{a}|{rel}|{b}"),
            "source": node_by_term[a]["id"],
            "target": node_by_term[b]["id"],
            "relationship_type": rel,
            "explanation": f"The source passage explicitly uses a relation cue consistent with '{rel}'.",
            "strength": round(strength, 3),
            "evidence_status": "validated_relation_cue",
            "evidence": rec["evidence"][:5],
        })

    return {
        "job_id": parsed.get("job_id"),
        "nodes": nodes,
        "edges": edges,
        "method": {
            "name": "semantic_passage_coverage_v1",
            "candidate_generation": "spaCy noun chunks/entities; lexical fallback if unavailable",
            "representation": "existing ResearchScope sentence embedder; cosine-normalized vectors",
            "assignment": f"Top semantic passage topics above cosine threshold {MIN_ASSIGNMENT_SIMILARITY}; maximum two topics per passage",
            "contribution": "Each assigned source passage contributes total mass 1, divided across assigned topics; topic mass normalized across retained topics",
            "relationship_validation": "Explicit relation cue plus both literal topic labels in one source sentence; semantic similarity alone never creates an edge",
            "percentage_semantics": "relative semantic passage coverage among retained topics; not word frequency, query similarity, or confidence",
        },
    }
