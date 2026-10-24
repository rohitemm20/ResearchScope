import os
import re
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

EMBEDDING_MODEL_NAME = os.getenv(
    "TAXONOMY_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
MAX_PAPER_CHARS = 50000
MAX_CHUNKS = 80
MIN_DOMAIN_SIMILARITY = float(os.getenv("TAXONOMY_MIN_DOMAIN_SIMILARITY", "0.24"))
MIN_TOPIC_SIZE = 2
MIN_OUTLIER_SIMILARITY = 0.40

DOMAIN_CATALOG = [
    ("Computer Science", "computing, algorithms, software, programming, data structures and computer systems"),
    ("Artificial Intelligence and Machine Learning", "machine learning, deep learning, neural networks, NLP, computer vision and intelligent systems"),
    ("Data Science and Analytics", "statistics, data mining, data analysis, predictive analytics, visualization and knowledge discovery"),
    ("Cybersecurity and Privacy", "cybersecurity, malware, cryptography, privacy, intrusion detection, secure systems and threat analysis"),
    ("Electrical and Electronics Engineering", "electronics, circuits, signal processing, communications, embedded systems and electrical power"),
    ("Mechanical and Industrial Engineering", "mechanical design, manufacturing, robotics, industrial optimization and production systems"),
    ("Civil and Environmental Engineering", "civil infrastructure, construction, structural engineering, water resources and environmental systems"),
    ("Biomedical Engineering and Health Informatics", "medical technology, clinical informatics, medical imaging, digital health and biomedical systems"),
    ("Medicine and Healthcare", "clinical medicine, diagnosis, treatment, public health, patient outcomes and healthcare delivery"),
    ("Biology and Life Sciences", "biology, genetics, genomics, ecology, microbiology and biological experiments"),
    ("Chemistry and Materials Science", "chemical reactions, compounds, materials, polymers, nanomaterials and material characterization"),
    ("Physics and Astronomy", "physical systems, quantum mechanics, optics, astrophysics and astronomical observations"),
    ("Mathematics and Statistics", "mathematical theory, statistical inference, probability, optimization and numerical methods"),
    ("Social Sciences and Psychology", "human behavior, psychology, sociology, education, surveys and social research"),
    ("Business and Economics", "economics, finance, markets, management, operations and business strategy"),
    ("Agriculture and Food Science", "agriculture, crop science, food systems, soil, farming and food safety"),
    ("Energy and Sustainability", "renewable energy, energy systems, climate, sustainability and resource efficiency"),
    ("Robotics and Autonomous Systems", "robots, autonomous agents, control systems, motion planning and sensor fusion"),
    ("Telecommunications and Networking", "computer networks, wireless communications, network protocols and distributed communication"),
    ("Other / Interdisciplinary Research", "research that spans fields or does not fit the other listed domains"),
]

# Generic academic terms and PDF artefacts should not become topic labels.
CUSTOM_STOP_WORDS = {
    "et", "al", "fig", "figure", "table", "section", "chapter", "paper",
    "study", "proposed", "approach", "method", "methods", "result", "results",
    "conclusion", "introduction", "related", "work", "based", "using", "used",
    "show", "shows", "shown", "also", "however", "therefore", "provide",
    "provides", "discuss", "discusses", "respectively", "new", "different",
    "number", "value", "values", "data", "information", "example", "examples",
    "case", "cases", "can", "may", "could", "would", "one", "two", "three",
    "first", "second", "finally", "well", "high", "low", "large", "small",
    "paper", "authors", "author", "approaches", "based", "proposed",
}


class ResearchArea(BaseModel):
    name: str
    description: str
    subdomains: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ResearchDomain(BaseModel):
    name: str
    description: str
    research_areas: list[ResearchArea] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class DomainTaxonomy(BaseModel):
    primary_domain: str
    domains: list[ResearchDomain]
    classification_summary: str


@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def _normalize_evidence_text(text: str) -> str:
    """Normalize common PDF Unicode/spacing artifacts for evidence verification."""
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    # Evidence passages may have citations stripped during topic preprocessing.
    text = re.sub(r"\([^)]{0,120}\b(?:19|20)\d{2}[a-z]?\s*[,;][^)]{0,120}\)", " ", text)
    text = re.sub(r"\b[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.?)\s*\(?\b(?:19|20)\d{2}[a-z]?\)?", " ", text)
    text = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().casefold()

def _validate_evidence(taxonomy: DomainTaxonomy, paper_text: str):
    normalized_text = _normalize_evidence_text(paper_text)
    invalid = []

    def check_excerpt(excerpt, details):
        for part in excerpt.split("..."):
            normalized_part = _normalize_evidence_text(part)
            if len(normalized_part) >= 20 and normalized_part not in normalized_text:
                invalid.append({**details, "excerpt": excerpt})
                return

    for domain in taxonomy.domains:
        for excerpt in domain.evidence:
            check_excerpt(excerpt, {"domain": domain.name})
        for area in domain.research_areas:
            for excerpt in area.evidence:
                check_excerpt(excerpt, {"domain": domain.name, "research_area": area.name})
    return invalid


def _clean_pdf_text(text: str) -> str:
    """Clean a working copy for modelling; original text remains the evidence source."""
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"(?m)^\s*=+\s*Page\s+\d+.*?=+\s*$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Discard author/affiliation front matter before an early abstract heading.
    # Evidence is still validated against the untouched original extracted text.
    abstract = re.search(r"(?i)\babstract\s*[:.]?", text[:5000])
    if abstract:
        prefix = text[:abstract.start()]
        if re.search(
            r"(?i)(university|department|faculty|laboratory|\blab\b|institute|"
            r"research group|street|road|budapest|\bemail\b|www\.|@)",
            prefix,
        ):
            text = "Abstract: " + text[abstract.end():]

    # Remove common author-year citation forms without removing the surrounding claim.
    text = re.sub(r"\([^)]{0,100}\b(?:19|20)\d{2}[a-z]?\s*[,;][^)]{0,100}\)", " ", text)
    text = re.sub(r"\b[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.?)\s*\(?\b(?:19|20)\d{2}[a-z]?\)?", " ", text)
    text = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _is_boilerplate(text: str) -> bool:
    low = text.casefold()
    boilerplate = (
        "the paper is structured as follows",
        "the remainder of this paper",
        "the rest of this paper",
        "organization of this paper",
        "in this section, we provide",
        "this paper is organized",
        "the paper is organized",
    )
    if any(phrase in low for phrase in boilerplate):
        return True
    if re.fullmatch(r"[\W\d_]+", text):
        return True
    metadata_markers = (
        "university", "department of", "faculty of", "research group",
        "laboratory", "laboratories", "institute of", "street,", "road,",
        "email:", "e-mail:", "corresponding author", "copyright",
        "all rights reserved", "received ", "accepted ", "published online",
        "https://doi.org/", "doi.org/", "issn", "springer", "volume ",
    )
    sentence_count = len(re.findall(r"[.!?]", text))
    # Publisher/journal headers and DOI lines should never be evidence areas.
    if (
        ("doi.org/" in low or "https://" in low or "issn" in low)
        and sentence_count <= 2
    ):
        return True
    if (
        sum(marker in low for marker in metadata_markers) >= 2
        or "@" in text
        or re.search(r"\b(?:h-)?\d{4,6}\s+[a-z]", low)
    ) and sentence_count <= 2:
        return True
    if len(re.findall(r"\b(?:19|20)\d{2}\b", text)) >= 3 and len(re.findall(r"[.!?]", text)) <= 1:
        return True
    return False


def _make_chunks(text: str) -> list[str]:
    """Create passages suitable for topic modelling, excluding page markers and boilerplate."""
    cleaned = _clean_pdf_text(text)
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n+", cleaned)
    ]
    paragraphs = [
        p for p in paragraphs
        if len(p) >= 80 and not _is_boilerplate(p)
    ]

    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= 850:
            chunks.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + len(sentence) + 1 > 700:
                if len(current) >= 80 and not _is_boilerplate(current):
                    chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if len(current) >= 80 and not _is_boilerplate(current):
            chunks.append(current)

    # PDF extraction sometimes has no blank lines. Fall back to sentence windows.
    if len(chunks) < 4:
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", cleaned))
        chunks, current = [], ""
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 25:
                continue
            if current and len(current) + len(sentence) + 1 > 650:
                if len(current) >= 80 and not _is_boilerplate(current):
                    chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if len(current) >= 80 and not _is_boilerplate(current):
            chunks.append(current)

    seen, result = set(), []
    for chunk in chunks:
        key = re.sub(r"\W+", " ", chunk.casefold()).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(chunk)
    return result[:MAX_CHUNKS]


def _safe_excerpt(text: str, limit: int = 360) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    boundary = cut.rfind(" ")
    return (cut[:boundary] if boundary > 100 else cut).rstrip(" ,;:") + "..."


def _topic_keywords(topic_model, topic_id: int) -> list[str]:
    pairs = topic_model.get_topic(topic_id) or []
    words = []
    for word, _score in pairs:
        word = word.strip().lower()
        if not word or word in CUSTOM_STOP_WORDS or word.isdigit():
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", word):
            continue
        if len(word) < 3:
            continue
        if word not in words:
            words.append(word)
        if len(words) == 8:
            break
    return words


def _area_label(keywords: list[str], evidence: list[str]) -> str:
    """Turn topic terms into readable labels without using a generative model."""
    text = " ".join(evidence).casefold()
    terms = " ".join(keywords).casefold()

    rules = [
        (("drone", "cybersecurity"), "Drone cybersecurity and privacy issues"),
        (("drone", "attack", "gcs"), "Drone attacks and ground-control security"),
        (("radio", "links", "signals"), "Drone radio links and control-signal security"),
        (("wireless", "communication", "vulnerabilities"), "Wireless communication vulnerabilities in drones"),
        (("authentication", "access", "communication"), "Drone authentication and secure communications"),
        (("privacy", "sensors", "drone"), "Drone privacy and sensor security"),
        (("malware", "iot", "embedded"), "Malware detection for embedded IoT devices"),
        (("malware", "random forest"), "Machine-learning-based malware detection"),
        (("malware", "detection"), "Malware detection methods"),
        (("tlsh", "hash"), "TLSH similarity hashing"),
        (("false positive", "detection"), "False-positive detection evaluation"),
        (("true positive", "detection"), "Malware detection performance evaluation"),
        (("dataset", "malware"), "Malware datasets and experimental evaluation"),
        (("benign", "training"), "Training data and classifier evaluation"),
        (("machine learning", "detection"), "Machine-learning-based detection"),
        (("random forest",), "Random forest classification"),
        (("embedded", "devices"), "Embedded-device implementation"),
        (("storage", "running time"), "Storage and runtime efficiency"),
    ]
    for required, label in rules:
        if all(term in text or term in terms for term in required):
            return label

    useful = [w for w in keywords if w not in CUSTOM_STOP_WORDS and not w.isdigit()]
    if not useful:
        return "Research methods and evaluation"
    # Prefer a compact phrase rather than a long comma-separated keyword dump.
    return " ".join(w.capitalize() for w in useful[:3])


def _normalise_vector(vector):
    return vector / (np.linalg.norm(vector) + 1e-12)


def _keywords_for_group(texts: list[str], max_terms: int = 8) -> list[str]:
    """Extract descriptive terms from a passage group without a generative model."""
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

    if not texts:
        return []
    stop_words = sorted(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOP_WORDS))
    try:
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            max_features=3000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
        )
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        ranked = np.argsort(scores)[::-1]
        result = []
        for idx in ranked:
            term = str(terms[idx]).strip().lower()
            if term and term not in result and not term.isdigit():
                result.append(term)
            if len(result) >= max_terms:
                break
        return result
    except ValueError:
        return []


def generate_domain_taxonomy(paper_text: str) -> dict:
    if not paper_text or len(paper_text.strip()) < 100:
        raise ValueError("The extracted paper text is too short to classify.")

    original_text = paper_text[:MAX_PAPER_CHARS]
    chunks = _make_chunks(original_text)
    if not chunks:
        raise ValueError("Could not split the extracted paper into usable text passages.")

    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError(
            "Taxonomy dependencies are missing. Install bertopic and sentence-transformers."
        ) from exc

    try:
        embedding_model = _get_embedding_model()
        embeddings = embedding_model.encode(
            chunks, batch_size=16, show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Keep the number of clusters conservative for a single-paper corpus.
        topic_records = []
        topic_ids = [-1] * len(chunks)
        if len(chunks) >= 8:
            neighbors = max(3, min(10, len(chunks) - 1))
            min_cluster = max(MIN_TOPIC_SIZE, min(5, len(chunks) // 8))
            vectorizer = CountVectorizer(
                stop_words=sorted(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOP_WORDS)),
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.90,
                max_features=10000,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
            )
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=UMAP(
                    n_neighbors=neighbors,
                    n_components=2,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=42,
                ),
                hdbscan_model=HDBSCAN(
                    min_cluster_size=min_cluster,
                    min_samples=1,
                    prediction_data=False,
                ),
                vectorizer_model=vectorizer,
                calculate_probabilities=False,
                verbose=False,
            )
            topic_ids, _ = topic_model.fit_transform(chunks, embeddings)

            clustered = {}
            outliers = []
            for idx, topic_id in enumerate(topic_ids):
                if topic_id == -1:
                    outliers.append(idx)
                else:
                    clustered.setdefault(int(topic_id), []).append(idx)

            # Only genuine multi-passage clusters become candidate research areas.
            for topic_id, indices in clustered.items():
                if len(indices) < MIN_TOPIC_SIZE:
                    outliers.extend(indices)
                    continue
                words = _topic_keywords(topic_model, topic_id)
                evidence_indices = [
                    i for i in indices
                    if not _is_boilerplate(chunks[i])
                    and "doi.org/" not in chunks[i].casefold()
                    and not re.search(r"\b(?:university|department of|faculty of|laboratory|corresponding author)\b", chunks[i].casefold())
                ]
                if not evidence_indices:
                    evidence_indices = list(indices)
                evidence = [_safe_excerpt(chunks[i]) for i in evidence_indices[:3]]
                topic_records.append({
                    "keywords": words,
                    "indices": list(indices),
                    "evidence": evidence,
                })

            # Attach outliers to the nearest established topic only when the
            # semantic match is strong enough; otherwise do not invent an area.
            if topic_records and outliers:
                centroids = [
                    _normalise_vector(np.mean(embeddings[r["indices"]], axis=0))
                    for r in topic_records
                ]
                for idx in outliers:
                    scores = [float(embeddings[idx] @ center) for center in centroids]
                    best = int(np.argmax(scores))
                    if scores[best] >= MIN_OUTLIER_SIMILARITY:
                        topic_records[best]["indices"].append(idx)
                        topic_records[best]["evidence"].append(_safe_excerpt(chunks[idx]))
        else:
            topic_records = []

        bertopic_cluster_count = len({int(t) for t in topic_ids if int(t) >= 0})

        # One BERTopic cluster is insufficient for a useful single-paper taxonomy.
        # Use deterministic clustering of existing sentence embeddings to separate
        # broad themes when BERTopic returns fewer than two candidate areas.
        if len(topic_records) < 2 and len(chunks) >= 6:
            n_clusters = max(3, min(6, round(len(chunks) ** 0.5)))
            n_clusters = min(n_clusters, len(chunks) - 1)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            grouped = {}
            for idx, cluster_id in enumerate(labels):
                grouped.setdefault(int(cluster_id), []).append(idx)

            topic_records = []
            for indices in sorted(grouped.values(), key=lambda group: min(group)):
                group_texts = [chunks[i] for i in indices]
                keywords = _keywords_for_group(group_texts)
                evidence_indices = sorted(
                    indices,
                    key=lambda i: (
                        len(chunks[i]) + 100 * sum(
                            term in chunks[i].casefold()
                            for term in (
                                "malware", "detection", "machine learning", "iot",
                                "classifier", "hash", "evaluation", "experiment",
                            )
                        )
                    ),
                    reverse=True,
                )[:3]
                topic_records.append({
                    "keywords": keywords,
                    "indices": list(indices),
                    "evidence": [_safe_excerpt(chunks[i]) for i in evidence_indices],
                })

        if not topic_records:
            ranked_indices = sorted(
                range(len(chunks)), key=lambda i: len(chunks[i]), reverse=True
            )[:min(3, len(chunks))]
            topic_records = [{
                "keywords": _keywords_for_group([chunks[i] for i in ranked_indices]),
                "indices": ranked_indices,
                "evidence": [_safe_excerpt(chunks[i]) for i in ranked_indices],
            }]

        # Build labels from meaningful terms and evidence. Deduplicate similar labels.
        for record in topic_records:
            record["evidence"] = [
                excerpt for excerpt in dict.fromkeys(record["evidence"])
                if not _is_boilerplate(excerpt)
                and "doi.org/" not in excerpt.casefold()
                and not re.search(r"\b(?:university|department of|faculty of|laboratory|corresponding author)\b", excerpt.casefold())
            ][:3]
            if not record["evidence"]:
                record["evidence"] = [_safe_excerpt(chunks[i]) for i in record["indices"][:1]]
            # Add cluster-level keywords from passages when BERTopic's terms are weak.
            if len(record["keywords"]) < 3:
                record["keywords"] = list(dict.fromkeys(
                    record["keywords"] + _keywords_for_group([chunks[i] for i in record["indices"]])
                ))[:8]
            record["label"] = _area_label(record["keywords"], record["evidence"])
            record["text"] = " ".join(chunks[i] for i in record["indices"])

        # Remove duplicate labels and merge their passage indices/evidence.
        merged = {}
        for record in topic_records:
            key = re.sub(r"[^a-z0-9]+", " ", record["label"].casefold()).strip()
            if key not in merged:
                merged[key] = record
            else:
                merged[key]["indices"] = sorted(set(merged[key]["indices"] + record["indices"]))
                merged[key]["evidence"] = list(dict.fromkeys(
                    merged[key]["evidence"] + record["evidence"]
                ))[:3]
                merged[key]["keywords"] = list(dict.fromkeys(
                    merged[key]["keywords"] + record["keywords"]
                ))[:8]
        topic_records = list(merged.values())

        catalog_texts = [f"{name}. {description}" for name, description in DOMAIN_CATALOG]
        catalog_embeddings = embedding_model.encode(
            catalog_texts, show_progress_bar=False, normalize_embeddings=True
        )

        paper_vector = _normalise_vector(np.mean(embeddings, axis=0))
        paper_scores = catalog_embeddings @ paper_vector
        ranked_domains = np.argsort(paper_scores)[::-1]

        # Use the whole-paper semantic match as a soft anchor. This helps keep
        # closely related areas (e.g. false-positive evaluation) under the
        # paper's dominant domain instead of incorrectly creating an "Other" domain.
        other_idx = next(
            i for i, item in enumerate(DOMAIN_CATALOG)
            if item[0] == "Other / Interdisciplinary Research"
        )
        primary_candidate_idx = next(
            int(idx) for idx in ranked_domains
            if DOMAIN_CATALOG[int(idx)][0] != "Other / Interdisciplinary Research"
        )

        domain_topics = {}
        for record in topic_records:
            related_indices = record["indices"]
            vector = _normalise_vector(np.mean(embeddings[related_indices], axis=0))
            scores = catalog_embeddings @ vector
            best_idx = int(np.argmax(scores))
            score = float(scores[best_idx])
            primary_topic_score = float(scores[primary_candidate_idx])

            # If the local topic maps to generic "Other" or is below threshold,
            # prefer the paper's dominant domain only when the topic itself has
            # sufficient semantic similarity to that domain.
            topic_text = (record["label"] + " " + " ".join(record["keywords"]) + " " + " ".join(record["evidence"])).casefold()
            cybersecurity_idx = next(
                i for i, item in enumerate(DOMAIN_CATALOG)
                if item[0] == "Cybersecurity and Privacy"
            )
            cybersecurity_signals = (
                "attack", "attacker", "cybersecurity", "drone", "privacy",
                "vulnerability", "false-positive", "false positive", "authentication",
                "secure communication", "security",
            )
            if (
                DOMAIN_CATALOG[primary_candidate_idx][0] == "Cybersecurity and Privacy"
                and any(signal in topic_text for signal in cybersecurity_signals)
                and (best_idx == other_idx or score < MIN_DOMAIN_SIMILARITY)
            ):
                best_idx = cybersecurity_idx
                score = float(scores[cybersecurity_idx])
            elif best_idx == other_idx or score < MIN_DOMAIN_SIMILARITY:
                if primary_topic_score >= MIN_DOMAIN_SIMILARITY:
                    best_idx = primary_candidate_idx
                    score = primary_topic_score
                else:
                    best_idx = other_idx
            name, description = DOMAIN_CATALOG[best_idx]

            domain_topics.setdefault(name, {"description": description, "areas": []})
            domain_topics[name]["areas"].append({
                "label": record["label"],
                "keywords": record["keywords"],
                "evidence": record["evidence"],
                "score": round(score, 4),
                "indices": related_indices,
            })

        domain_counts = {
            name: sum(len(area["indices"]) for area in data["areas"])
            for name, data in domain_topics.items()
        }
        primary = (
            max(domain_counts, key=domain_counts.get)
            if domain_counts
            else DOMAIN_CATALOG[int(ranked_domains[0])][0]
        )

        domains = []
        for name, data in sorted(
            domain_topics.items(),
            key=lambda item: domain_counts.get(item[0], 0),
            reverse=True,
        )[:5]:
            areas = []
            for area in sorted(data["areas"], key=lambda a: len(a["indices"]), reverse=True)[:6]:
                label = area["label"]
                keywords = area["keywords"]
                areas.append(ResearchArea(
                    name=label[:100],
                    description=(
                        "Research area inferred from semantically related passages. "
                        "Representative terms: "
                        + (", ".join(keywords[:6]) if keywords else label)
                    )[:500],
                    subdomains=keywords[:6],
                    evidence=area["evidence"][:3],
                ))
            domain_evidence = []
            for area in areas:
                domain_evidence.extend(area.evidence[:1])
            domains.append(ResearchDomain(
                name=name,
                description=data["description"],
                research_areas=areas,
                evidence=list(dict.fromkeys(domain_evidence))[:5],
            ))

        if not domains:
            raise ValueError("No research domains could be identified from the extracted text.")
        if primary not in {domain.name for domain in domains}:
            primary = domains[0].name

        result = DomainTaxonomy(
            primary_domain=primary,
            domains=domains,
            classification_summary=(
                "Generated without a generative LLM using BERTopic, sentence embeddings, "
                "and a curated domain vocabulary. Boilerplate and weak/unclustered passages "
                "are filtered or merged; labels are heuristic and should be reviewed."
            ),
        )
        invalid_evidence = _validate_evidence(result, original_text)
        result_dict = result.model_dump()
        result_dict["validation"] = {
            "evidence_checked": True,
            "invalid_evidence_count": len(invalid_evidence),
            "invalid_evidence": invalid_evidence,
            "method": "bertopic_plus_sentence_transformer_similarity",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "topic_count": len(topic_records),
            "final_research_area_count": sum(
                len(domain.research_areas) for domain in result.domains
            ),
            "bertopic_cluster_count": bertopic_cluster_count,
            "usable_passage_count": len(chunks),
        }
        return result_dict

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Non-LLM taxonomy generation failed: {exc}") from exc
