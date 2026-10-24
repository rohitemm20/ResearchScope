import os
import re
from functools import lru_cache
from typing import List

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


# Broad, curated labels are used only as an anchor vocabulary. Topic areas and
# subdomains are discovered from the current paper's own text.
DOMAIN_CATALOG = [
    ("Computer Science", "computing, algorithms, software, programming, data structures and computer systems"),
    ("Artificial Intelligence and Machine Learning", "machine learning, deep learning, neural networks, generative models, NLP, computer vision and intelligent systems"),
    ("Data Science and Analytics", "statistics, data mining, data analysis, predictive analytics, visualization and knowledge discovery"),
    ("Cybersecurity and Privacy", "cybersecurity, cryptography, privacy, intrusion detection, secure systems and threat analysis"),
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


@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def _normalize_evidence_text(text: str) -> str:
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
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


def _make_chunks(text: str) -> list[str]:
    """Prefer paragraphs, then split oversized paragraphs into sentence groups."""
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n+", text)]
    paragraphs = [p for p in paragraphs if len(p) >= 60]
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= 900:
            chunks.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > 750:
                if len(current) >= 60:
                    chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if len(current) >= 60:
            chunks.append(current)
    if len(chunks) < 5:
        # PDFs sometimes have no paragraph breaks; split on sentence boundaries.
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
        chunks = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > 650:
                if len(current) >= 60:
                    chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if len(current) >= 60:
            chunks.append(current)
    # Deduplicate near-identical repeated PDF lines and bound CPU/RAM use.
    seen, result = set(), []
    for chunk in chunks:
        key = chunk.casefold()
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result[:MAX_CHUNKS]


def _topic_keywords(topic_model, topic_id: int) -> list[str]:
    pairs = topic_model.get_topic(topic_id) or []
    return [word for word, _score in pairs[:8]]


def _safe_excerpt(text: str, limit: int = 360) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    boundary = cut.rfind(" ")
    return (cut[:boundary] if boundary > 100 else cut).rstrip(" ,;:") + "..."


def generate_domain_taxonomy(paper_text: str) -> dict:
    if not paper_text or len(paper_text.strip()) < 100:
        raise ValueError("The extracted paper text is too short to classify.")

    paper_text = paper_text[:MAX_PAPER_CHARS]
    chunks = _make_chunks(paper_text)
    if not chunks:
        raise ValueError("Could not split the extracted paper into usable text passages.")

    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        from hdbscan import HDBSCAN
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

        # BERTopic needs several documents for meaningful clustering. For very
        # short papers, semantic domain matching still works without clustering.
        topic_records = []
        if len(chunks) >= 6:
            neighbors = max(2, min(5, len(chunks) - 1))
            min_cluster = max(2, min(5, len(chunks) // 5))
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=UMAP(
                    n_neighbors=neighbors, n_components=2,
                    min_dist=0.0, metric="cosine", random_state=42,
                ),
                hdbscan_model=HDBSCAN(
                    min_cluster_size=min_cluster,
                    min_samples=1,
                    prediction_data=False,
                ),
                vectorizer_model=CountVectorizer(
                    stop_words="english", ngram_range=(1, 2),
                    min_df=1, max_features=12000,
                ),
                calculate_probabilities=False,
                verbose=False,
            )
            topic_ids, _ = topic_model.fit_transform(chunks, embeddings)
            for topic_id in sorted(set(topic_ids)):
                indices = [i for i, value in enumerate(topic_ids) if value == topic_id]
                # Treat BERTopic's -1 outlier bucket as individual passages so
                # potentially useful evidence is not discarded.
                groups = [[i] for i in indices] if topic_id == -1 else [indices]
                for group in groups:
                    words = _topic_keywords(topic_model, topic_id) if topic_id != -1 else []
                    sample_indices = sorted(group, key=lambda i: -len(chunks[i]))[:3]
                    evidence = [_safe_excerpt(chunks[i]) for i in sample_indices]
                    label = ", ".join(words[:4]) if words else "Unclustered research passage"
                    topic_records.append({
                        "label": label,
                        "keywords": words,
                        "indices": group,
                        "evidence": evidence,
                        "text": " ".join(chunks[i] for i in group[:5]),
                    })
        else:
            # Short papers do not provide enough samples for reliable HDBSCAN.
            # Use their passages directly for taxonomy mapping.
            for i, chunk in enumerate(chunks):
                topic_records.append({
                    "label": "Research passage",
                    "keywords": [],
                    "indices": [i],
                    "evidence": [_safe_excerpt(chunk)],
                    "text": chunk,
                })

        catalog_texts = [f"{name}. {description}" for name, description in DOMAIN_CATALOG]
        catalog_embeddings = embedding_model.encode(
            catalog_texts, show_progress_bar=False, normalize_embeddings=True
        )
        import numpy as np

        # Pool the paper passage embeddings into one vector for primary domain.
        paper_vector = np.mean(embeddings, axis=0)
        paper_vector = paper_vector / (np.linalg.norm(paper_vector) + 1e-12)
        paper_scores = catalog_embeddings @ paper_vector
        ranked_domains = np.argsort(paper_scores)[::-1]

        # Assign each discovered topic to its best matching taxonomy anchor.
        domain_topics = {}
        for record in topic_records:
            related_indices = record["indices"]
            vector = np.mean(embeddings[related_indices], axis=0)
            vector = vector / (np.linalg.norm(vector) + 1e-12)
            scores = catalog_embeddings @ vector
            best_idx = int(np.argmax(scores))
            score = float(scores[best_idx])
            if score < MIN_DOMAIN_SIMILARITY:
                best_idx = next(
                    i for i, item in enumerate(DOMAIN_CATALOG)
                    if item[0] == "Other / Interdisciplinary Research"
                )
            name, description = DOMAIN_CATALOG[best_idx]
            domain_topics.setdefault(name, {"description": description, "areas": []})
            domain_topics[name]["areas"].append({
                "label": record["label"],
                "keywords": record["keywords"],
                "evidence": record["evidence"],
                "score": round(score, 4),
            })

        # Prefer domains with more mapped passages. Always ensure primary domain
        # is present in the returned domain list.
        domain_counts = {
            name: sum(len(area["evidence"]) for area in data["areas"])
            for name, data in domain_topics.items()
        }
        primary = max(domain_counts, key=domain_counts.get) if domain_counts else DOMAIN_CATALOG[int(ranked_domains[0])][0]

        domains = []
        for name, data in sorted(
            domain_topics.items(),
            key=lambda item: domain_counts.get(item[0], 0),
            reverse=True,
        )[:6]:
            areas = []
            for area in data["areas"][:8]:
                label = area["label"]
                keywords = area["keywords"]
                if label == "Research passage":
                    label = ", ".join(keywords[:4]) if keywords else "Research methods and findings"
                if label == "Unclustered research passage":
                    label = ", ".join(keywords[:4]) if keywords else "Unclustered research passage"
                # Keep a useful area name without using a text-generation model.
                areas.append(ResearchArea(
                    name=label[:100],
                    description=(
                        "Topic identified from representative terms: "
                        + (", ".join(keywords[:8]) if keywords else label)
                    )[:500],
                    subdomains=keywords[:6],
                    evidence=area["evidence"][:3],
                ))
            evidence = []
            for area in areas:
                evidence.extend(area.evidence[:1])
            domains.append(ResearchDomain(
                name=name,
                description=data["description"],
                research_areas=areas,
                evidence=list(dict.fromkeys(evidence))[:5],
            ))

        if not domains:
            raise ValueError("No research domains could be identified from the extracted text.")
        if primary not in {domain.name for domain in domains}:
            primary = domains[0].name

        result = DomainTaxonomy(
            primary_domain=primary,
            domains=domains,
            classification_summary=(
                "Generated without an LLM using BERTopic topic discovery and "
                "sentence-embedding similarity against a curated domain vocabulary. "
                "Review the suggested labels and evidence; similarity is a heuristic, not a calibrated probability."
            ),
        )
        invalid_evidence = _validate_evidence(result, paper_text)
        result_dict = result.model_dump()
        result_dict["validation"] = {
            "evidence_checked": True,
            "invalid_evidence_count": len(invalid_evidence),
            "invalid_evidence": invalid_evidence,
            "method": "bertopic_plus_sentence_transformer_similarity",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "topic_count": len(topic_records),
        }
        return result_dict

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Non-LLM taxonomy generation failed: {exc}") from exc
