from __future__ import annotations

import hashlib

import re

from collections import Counter, defaultdict

from functools import lru_cache

from typing import Any

import numpy as np

from .nlp_utils import split_sentences

from .text_cleaning import clean_document

METHOD_NAME = "semantic_passage_coverage_v3"

MAX_TOPICS = 12

MIN_TOPIC_SUPPORT = 2

MIN_ASSIGNMENT_SIMILARITY = 0.22

TOPIC_MERGE_SIMILARITY = 0.90

MAX_CANDIDATES = 100

MAX_MENTION_GAP_CHARS = 220

MAX_PRE_CUE_CHARS = 60   # text between left mention and the cue

MAX_POST_CUE_CHARS = 45  # text between the cue and right mention

MAX_TOPICS_PER_PASSAGE = 2

# (type, regex, reversed, symmetric)

# reversed: the grammatical subject comes *after* the cue ("B is used by A"),

#           so the edge points from the later mention to the earlier one.

# Extend only with tested, domain-safe patterns.

RELATION_CUES: list[tuple[str, re.Pattern, bool, bool]] = [

    ("evaluated_by", re.compile(r"\b(?:evaluated|tested|measured|assessed|validated)\s+(?:using|with|by|against|on)\b", re.I), False, False),

    ("improves", re.compile(r"\b(?:improves|enhances|increases|boosts|reduces|mitigates|outperforms)\b", re.I), False, False),

    ("improves", re.compile(r"\b(?:improved|enhanced|boosted|reduced|mitigated)\s+by\b", re.I), True, False),

    ("uses", re.compile(r"\b(?:uses|using|employs|leverages|adopts|applies|utili[sz]es)\b", re.I), False, False),

    ("uses", re.compile(r"\b(?:used|employed|applied|leveraged|adopted)\s+(?:by|in)\b", re.I), True, False),

    ("depends_on", re.compile(r"\b(?:depends on|is based on|are based on|relies on|requires)\b", re.I), False, False),

    ("compared_with", re.compile(r"\b(?:compared\s+(?:with|to|against)|in contrast to|versus|vs\.)\b", re.I), False, True),

    ("produces", re.compile(r"\b(?:produces|generates|predicts|classifies|detects|outputs)\b", re.I), False, False),

    ("produces", re.compile(r"\b(?:produced|generated|predicted|classified|detected)\s+by\b", re.I), True, False),

]

EDGE_LABELS = {

    "evaluated_by": "is evaluated with",

    "improves": "improves",

    "uses": "uses",

    "depends_on": "depends on",

    "compared_with": "is compared with",
    "co_occurs_with": "co-occurs with",

    "produces": "produces",

}

def _stable_id(prefix: str, value: str) -> str:

    return prefix + "_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]

def _normal(text: str) -> str:

    return re.sub(r"\s+", " ", text.lower()).strip()

_LEADING_NOISE = {

    "the", "a", "an", "new", "novel", "proposed", "previously", "recent",

    "various", "several", "different", "following", "using", "based",

    "results", "result", "approach", "method", "methods", "system", "systems",

    "this", "that", "these", "those", "our", "their", "its", "such", "each",

    "both", "many", "some", "other", "more", "most", "all", "any",

}

_GENERIC_ONLY = {

    "paper", "study", "work", "result", "results", "approach", "method",

    "methods", "system", "systems", "model", "models", "technique", "techniques",

    "process", "processes", "thing", "things", "value", "values", "way", "ways",

    "case", "cases", "number", "set", "sets", "type", "types", "kind", "part",

    "section", "figure", "table", "example", "examples", "order", "terms",

    "use", "time", "level", "levels", "form", "fact", "one", "two", "three",

}

_ACRONYMS = {

    "ai": "AI", "ml": "ML", "iot": "IoT", "tls": "TLS", "tlsh": "TLSH",

    "api": "API", "apis": "APIs", "cnn": "CNN", "cnns": "CNNs",

    "rnn": "RNN", "rnns": "RNNs", "svm": "SVM", "svms": "SVMs",

    "nlp": "NLP", "llm": "LLM", "llms": "LLMs", "gpu": "GPU", "cpu": "CPU",

    "tcp": "TCP", "ip": "IP", "http": "HTTP", "https": "HTTPS",

    "dns": "DNS", "dos": "DoS", "ddos": "DDoS", "gnn": "GNN", "gnns": "GNNs",

    "lstm": "LSTM", "gan": "GAN", "gans": "GANs", "bert": "BERT",

    "ids": "IDS", "vpn": "VPN", "sql": "SQL", "xml": "XML", "json": "JSON",

}

# spaCy entity labels that never make useful research topics.

_BAD_ENT_LABELS = {

    "CARDINAL", "ORDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",

}

_VERB_RE = re.compile(

    r"\b(?:is|are|was|were|has|have|had|can|could|would|should|will|may|might|do|does|did)\b")

def _singular(word: str) -> str:

    """Very small plural folding; good enough to merge 'networks'/'network'."""

    if len(word) > 4 and word.endswith("ies"):

        return word[:-3] + "y"

    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):

        return word[:-1]

    return word

def _fold(term: str) -> str:

    return " ".join(_singular(w) for w in term.split())

def _clean_candidate(phrase: str) -> str | None:

    """Normalize a candidate phrase and reject fragments/generic-only labels."""

    phrase = phrase.replace("\u2019", "'")

    phrase = re.sub(r"'s\b", "", phrase, flags=re.I)

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

    if _VERB_RE.search(normalized):

        return None

    if len(normalized) < 3 or re.fullmatch(r"[\d.\s]+", normalized):

        return None

    return normalized

def _display_label(term: str) -> str:

    """Human-readable label; keep common technical acronyms recognizable."""

    shown = []

    for word in term.split():

        key = word.lower().strip(".,:;()[]{}")

        shown.append(_ACRONYMS.get(key, word if word.isupper() else word.capitalize()))

    return " ".join(shown)

@lru_cache(maxsize=1)

def _get_spacy():

    """Load spaCy once. Returns (nlp, has_parser) or (None, False)."""

    try:

        import spacy

    except Exception:

        return None, False

    try:

        nlp = spacy.load("en_core_web_sm")

        return nlp, True

    except OSError:

        nlp = spacy.blank("en")

        nlp.add_pipe("sentencizer")

        return nlp, False

def _get_embedder():

    from .pipeline import get_embedder  # lazy: pulls in the search stack

    return get_embedder()

_STOP = set("""the a an and or but for with from into over under using use based

this that these those their its our we they is are was were be been being to of

in on as by at it can may might will would should have has had which also than

then there here when where while such not no more most other some any each both""".split())


def _topic_candidates(sentences: list[dict[str, Any]]) -> dict[str, int]:
    """Cleaned noun-phrase/entity candidates -> frequency (plural-folded)."""
    raw: Counter[str] = Counter()

    nlp, has_parser = _get_spacy()

    if nlp is not None:
        texts = [item["text"][:5000] for item in sentences]
        try:
            for doc in nlp.pipe(texts, batch_size=64):
                spans = (
                    list(doc.noun_chunks)
                    if has_parser and doc.has_annotation("DEP")
                    else []
                )
                spans.extend(
                    e for e in doc.ents
                    if e.label_ not in _BAD_ENT_LABELS
                )

                for span in spans:
                    cleaned = _clean_candidate(span.text)
                    if cleaned:
                        raw[cleaned] += 1
        except Exception:
            raw.clear()

    if not raw:  # lexical fallback; same cleaning rules
        for item in sentences:
            words = re.findall(
                r"\b[A-Za-z][A-Za-z0-9.+_-]{2,}\b",
                item["text"],
            )

            for word in words:
                if word.lower() not in _STOP:
                    cleaned = _clean_candidate(word)
                    if cleaned:
                        raw[cleaned] += 1

            for a, b in zip(words, words[1:]):
                if a.lower() not in _STOP and b.lower() not in _STOP:
                    cleaned = _clean_candidate(f"{a} {b}")
                    if cleaned:
                        raw[cleaned] += 1

    # Fold plurals: keep the most frequent surface form as canonical.
    groups: dict[str, list[str]] = defaultdict(list)

    for term in raw:
        groups[_fold(term)].append(term)

    folded: dict[str, int] = {}

    for variants in groups.values():
        canonical = max(
            variants,
            key=lambda t: (raw[t], -len(t), t),
        )
        folded[canonical] = sum(raw[v] for v in variants)

    ranked = sorted(folded, key=lambda t: (-folded[t], t))
    kept: list[str] = []

    for term in ranked:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

        has_matching_longer_term = False

        for other in kept:
            if re.search(pattern, other) and folded[other] >= folded[term]:
                has_matching_longer_term = True
                break

        if has_matching_longer_term:
            continue  # fragment of a more frequent longer phrase

        kept.append(term)

        if len(kept) >= MAX_CANDIDATES:
            break

    return {t: folded[t] for t in kept}


def _empty(parsed: dict[str, Any], message: str) -> dict[str, Any]:

    return {

        "job_id": parsed.get("job_id"), "nodes": [], "edges": [],

        "method": {"name": METHOD_NAME}, "message": message,

    }

def _normalise_rows(m: np.ndarray) -> np.ndarray:

    m = np.asarray(m, dtype=float)

    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

def _drop_near_duplicates(terms: list[str], vecs: np.ndarray) -> tuple[list[str], np.ndarray]:

    """Greedy: keep the higher-ranked term when two are almost the same idea."""

    sims = vecs @ vecs.T

    keep: list[int] = []

    for i in range(len(terms)):

        if all(sims[i, j] < TOPIC_MERGE_SIMILARITY for j in keep):

            keep.append(i)

    return [terms[i] for i in keep], vecs[keep]


def _find_mentions(
    text: str,
    terms: list[str],
) -> list[tuple[int, int, str]]:
    """Non-overlapping literal mentions, longest terms first, in reading order."""
    low = text.lower()
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, int, str]] = []

    for term in sorted(terms, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(term) + r"(?:s|es)?(?!\w)"

        for match in re.finditer(pattern, low):
            start, end = match.span()

            if any(start < te and end > ts for ts, te in taken):
                continue

            taken.append((start, end))
            found.append((start, end, term))

    found.sort()
    return found

def _relation_between(gap: str) -> tuple[str, re.Pattern, bool, bool, str] | None:

    """First relation cue found in the text *between* two mentions."""

    best = None

    for rel, rx, rev, sym in RELATION_CUES:

        m = rx.search(gap)

        if not m:

            continue

        # The cue must hug both mentions. Extra words between a cue and a

        # mention usually mean an untracked noun sits in the middle

        # ("produced by static analysis of X"), so the link is not direct.

        if m.start() > MAX_PRE_CUE_CHARS or len(gap) - m.end() > MAX_POST_CUE_CHARS:

            continue

        if best is None or m.start() < best[0]:

            best = (m.start(), rel, rx, rev, sym, m.group(0))

    if best is None:

        return None

    _, rel, rx, rev, sym, cue = best

    return rel, rx, rev, sym, cue

def build_topic_network(parsed: dict[str, Any], max_topics: int = MAX_TOPICS) -> dict[str, Any]:

    paragraphs = clean_document(parsed)

    sentence_rows = split_sentences(paragraphs)

    if not sentence_rows:

        return _empty(parsed, "No usable extracted text was available.")

    sentences = [{

        "id": str(row.get("id", i)),

        "text": row["text"],

        "page": row.get("page"),

        "section": row.get("section"),

        "block_ids": row.get("block_ids", []),

    } for i, row in enumerate(sentence_rows)]

    freq = _topic_candidates(sentences)

    if not freq:

        return _empty(parsed, "No stable topic candidates could be extracted.")

    candidates = list(freq)

    embedder = _get_embedder()

    passage_vecs = _normalise_rows(embedder.encode([s["text"] for s in sentences]))

    topic_vecs = _normalise_rows(embedder.encode(candidates))

    candidates, topic_vecs = _drop_near_duplicates(candidates, topic_vecs)

    index_of = {t: i for i, t in enumerate(candidates)}

    similarities = passage_vecs @ topic_vecs.T  # (passages, topics)

    # Vectorised top-k per passage.

    k = min(MAX_TOPICS_PER_PASSAGE, similarities.shape[1])

    top_idx = np.argsort(-similarities, axis=1)[:, :k]

    support: dict[str, list[tuple[int, float]]] = defaultdict(list)

    raw_contribution: dict[str, float] = defaultdict(float)

    for i in range(len(sentences)):

        assigned = [

            int(j) for j in top_idx[i]

            if similarities[i, j] >= MIN_ASSIGNMENT_SIMILARITY

        ]

        if not assigned:

            continue

        weight = 1.0 / len(assigned)  # each passage carries total mass 1

        for j in assigned:

            term = candidates[j]

            support[term].append((i, float(similarities[i, j])))

            raw_contribution[term] += weight

    def ranked(pool):

        return sorted(pool, key=lambda t: (-raw_contribution[t], t))[:max_topics]

    viable = ranked(t for t in candidates if len(support[t]) >= MIN_TOPIC_SUPPORT)

    if not viable:

        # Very short papers: allow single-passage support rather than invent topics.

        viable = ranked(t for t in candidates if support[t])

    # Use all assigned candidate-topic mass as the denominator. Topics excluded
    # from the displayed top-N remain in the denominator, preventing the
    # displayed subset from being artificially inflated to 100%.
    total = sum(raw_contribution.values())

    if total <= 0:
        return _empty(parsed, "Text was available but no topics met the support threshold.")

    exact = {t: raw_contribution[t] / total * 100.0 for t in viable}
    rounded = {t: round(exact[t], 2) for t in viable}

    # ---- nodes ----------------------------------------------------------

    nodes: list[dict[str, Any]] = []

    for term in viable:

        rows = sorted(support[term], key=lambda x: (-x[1], x[0]))

        evidence = [{

            "text": sentences[i]["text"],

            "page": sentences[i]["page"],

            "section": sentences[i]["section"],

            "block_ids": sentences[i]["block_ids"],

            "similarity": round(sim, 4),

            "status": "source_passage",

        } for i, sim in rows[:5]]

        sections = Counter(sentences[i]["section"] for i, _ in rows if sentences[i]["section"])

        nodes.append({

            "id": _stable_id("topic", term),

            "label": _display_label(term),

            "percentage": round(rounded[term], 2),

            "passage_support": len(rows),

            "mentions": sum(1 for s in sentences if _find_mentions(s["text"], [term])),

            "sections": [name for name, _ in sections.most_common(3)],

            "description": f"Supported by {len(rows)} semantically assigned source passage(s).",

            "evidence": evidence,

        })

    node_by_term = dict(zip(viable, nodes))

    # ---- edges ----------------------------------------------------------

    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    for sent in sentences:

        mentions = _find_mentions(sent["text"], viable)

        for (s1, e1, left), (s2, e2, right) in zip(mentions, mentions[1:]):

            if left == right or s2 - e1 > MAX_MENTION_GAP_CHARS:

                continue

            gap = sent["text"][e1:s2]

            hit = _relation_between(gap)
            if hit:
                relation, _, reverse, symmetric, cue = hit
            else:
                # Evidence-backed co-occurrence is shown as an undirected edge,
                # not as a causal or directional relationship.
                relation, reverse, symmetric, cue = "co_occurs_with", False, True, None

            source, target = (right, left) if reverse else (left, right)
            if symmetric:
                source, target = sorted((source, target))

            key = (source, target, relation)

            rec = edges_by_key.setdefault(key, {"symmetric": symmetric, "evidence": []})

            rec["evidence"].append({

                "text": sent["text"], "page": sent["page"],

                "section": sent["section"], "block_ids": sent["block_ids"],

                "status": "validated_relation_cue" if cue else "co_occurrence",
                "cue": cue,

            })

    edges: list[dict[str, Any]] = []

    degree: Counter[str] = Counter()

    for (source, target, relation), rec in sorted(edges_by_key.items()):

        unique = {e["text"] for e in rec["evidence"]}

        # Structural diagnostic only: distinct evidence sentences, capped at 3.

        strength = min(1.0, len(unique) / 3.0)

        s_node, t_node = node_by_term[source], node_by_term[target]

        degree[s_node["id"]] += 1

        degree[t_node["id"]] += 1

        edges.append({

            "id": _stable_id("edge", f"{source}|{relation}|{target}"),

            "source": s_node["id"],

            "target": t_node["id"],

            "relationship_type": relation,

            "relationship_label": EDGE_LABELS.get(relation, relation.replace("_", " ")),

            "directed": not rec["symmetric"],

            "explanation": (

                f"{s_node['label']} {EDGE_LABELS.get(relation, relation)} {t_node['label']}, "

                "based on an explicit cue in the source text."

            ),

            "strength": round(strength, 3),

            "evidence_status": "validated_relation_cue",

            "evidence": rec["evidence"][:5],

        })

    for node in nodes:

        node["degree"] = degree.get(node["id"], 0)

    return {

        "job_id": parsed.get("job_id"),

        "nodes": nodes,

        "edges": edges,

        "stats": {

            "passages": len(sentences),

            "candidates": len(candidates),

            "topics": len(nodes),

            "edges": len(edges),

        },

        "method": {

            "name": METHOD_NAME,

            "candidate_generation": "spaCy noun chunks/entities (numbers and dates excluded); plural folding; lexical fallback if spaCy is unavailable",

            "representation": "existing ResearchScope sentence embedder; cosine-normalized vectors",

            "deduplication": f"Candidates with cosine similarity >= {TOPIC_MERGE_SIMILARITY} to a higher-ranked candidate are merged",

            "assignment": f"Top semantic passage topics above cosine threshold {MIN_ASSIGNMENT_SIMILARITY}; maximum {MAX_TOPICS_PER_PASSAGE} topics per passage",

            "contribution": "Each assigned source passage contributes total mass 1, divided across assigned topics; displayed topic percentages use all assigned candidate-topic mass as the denominator",

            "relationship_validation": "Explicit relation cues create typed edges; adjacent retained topics in the same source sentence may also receive an undirected co-occurrence edge, which does not imply causation",

            "percentage_semantics": "each displayed topic contribution as a percentage of all assigned candidate-topic mass; not word frequency, query similarity, or confidence",

        },

    }
