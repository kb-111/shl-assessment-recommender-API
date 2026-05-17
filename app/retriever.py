"""
Retrieval layer: semantic search via FAISS + metadata-aware reranking.

Design decisions:
  - Loads index + metadata once at startup (module-level singleton).
  - Exposes retrieve(query, constraints, top_k) as the single public function.
  - Reranking boosts items that match structured constraints (test_type, job level, etc.)
  - Never invents items — every returned dict comes verbatim from catalog metadata.
"""

import json
import logging
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models import CatalogItem

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once per process)
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_catalog: list[dict] = []


def _load_resources() -> None:
    global _model, _index, _catalog

    if _model is not None:
        return  # already loaded

    log.info("Loading embedding model…")
    _model = SentenceTransformer(settings.embedding_model)

    index_path = Path(settings.faiss_index_path)
    meta_path = Path(settings.faiss_meta_path)

    if index_path.exists() and meta_path.exists():
        log.info("Loading FAISS index from %s", index_path)
        _index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            _catalog = json.load(f)
        log.info("Index loaded: %d items", len(_catalog))
    else:
        # Fallback: build from catalog JSON at runtime (slower but works on Render)
        catalog_path = Path(settings.catalog_path)
        if not catalog_path.exists():
            raise RuntimeError(
                f"Neither FAISS index ({index_path}) nor catalog ({catalog_path}) found. "
                "Run scripts/scraper.py then scripts/build_index.py first."
            )
        log.warning("FAISS index not found — building in memory from catalog JSON…")
        _build_index_in_memory(catalog_path)


def _build_text(item: dict) -> str:
    parts = [item.get("name", "")]
    desc = item.get("description", "")
    if desc:
        parts.append(desc[:300])
    skills = item.get("skills", [])
    if skills:
        parts.append("skills: " + ", ".join(skills))
    levels = item.get("job_levels", [])
    if levels:
        parts.append("levels: " + ", ".join(levels))
    tt = item.get("test_type", "")
    type_map = {
        "A": "ability cognitive aptitude",
        "K": "knowledge skills technical",
        "P": "personality behaviour traits",
        "S": "situational judgement simulation",
        "C": "competency behavioural",
        "E": "exercise work sample",
        "B": "biodata",
    }
    if tt:
        parts.append(type_map.get(tt, ""))
    return " | ".join(p for p in parts if p)


def _build_index_in_memory(catalog_path: Path) -> None:
    global _index, _catalog
    with open(catalog_path, encoding="utf-8") as f:
        _catalog = json.load(f)

    texts = [_build_text(item) for item in _catalog]
    embeddings: np.ndarray = _model.encode(  # type: ignore[union-attr]
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings)
    log.info("Built in-memory FAISS index: %d items, dim=%d", len(_catalog), dim)


# ---------------------------------------------------------------------------
# Structured constraints extracted from conversation
# ---------------------------------------------------------------------------
class Constraints:
    """
    Structured intent extracted from conversation history by the LLM.
    Used for metadata-aware reranking on top of semantic similarity.
    """

    def __init__(
        self,
        role: str = "",
        seniority: str = "",
        technical_skills: list[str] | None = None,
        needs_personality: bool = False,
        needs_cognitive: bool = False,
        needs_situational: bool = False,
        needs_knowledge: bool = False,
        remote_required: bool = False,
        adaptive_required: bool = False,
        duration_max_minutes: int | None = None,
        languages: list[str] | None = None,
    ):
        self.role = role
        self.seniority = seniority
        self.technical_skills = technical_skills or []
        self.needs_personality = needs_personality
        self.needs_cognitive = needs_cognitive
        self.needs_situational = needs_situational
        self.needs_knowledge = needs_knowledge
        self.remote_required = remote_required
        self.adaptive_required = adaptive_required
        self.duration_max_minutes = duration_max_minutes
        self.languages = languages or []

    @classmethod
    def from_dict(cls, d: dict) -> "Constraints":
        return cls(
            role=d.get("role", ""),
            seniority=d.get("seniority", ""),
            technical_skills=d.get("technical_skills", []),
            needs_personality=d.get("needs_personality", False),
            needs_cognitive=d.get("needs_cognitive", False),
            needs_situational=d.get("needs_situational", False),
            needs_knowledge=d.get("needs_knowledge", False),
            remote_required=d.get("remote_required", False),
            adaptive_required=d.get("adaptive_required", False),
            duration_max_minutes=d.get("duration_max_minutes"),
            languages=d.get("languages", []),
        )

    def to_query_string(self) -> str:
        """Convert constraints to a search query."""
        parts = []
        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        if self.technical_skills:
            parts.append(" ".join(self.technical_skills))
        if self.needs_personality:
            parts.append("personality behaviour traits")
        if self.needs_cognitive:
            parts.append("cognitive ability aptitude reasoning")
        if self.needs_situational:
            parts.append("situational judgement")
        if self.needs_knowledge:
            parts.append("knowledge skills technical")
        return " ".join(parts) if parts else "assessment"


def _rerank_score(item: dict, constraints: Constraints, semantic_score: float) -> float:
    """
    Combine semantic score with metadata bonus terms.
    Returns a float where higher = more relevant.
    """
    score = semantic_score
    tt = item.get("test_type", "")

    # Boost by test type match
    if constraints.needs_personality and tt == "P":
        score += 0.15
    if constraints.needs_cognitive and tt == "A":
        score += 0.15
    if constraints.needs_situational and tt == "S":
        score += 0.12
    if constraints.needs_knowledge and tt == "K":
        score += 0.10

    # Remote/adaptive hard filter (demote rather than exclude for robustness)
    if constraints.remote_required and not item.get("remote_testing", False):
        score -= 0.2
    if constraints.adaptive_required and not item.get("adaptive", False):
        score -= 0.2

    # Seniority match via job_levels
    if constraints.seniority:
        seniority_lower = constraints.seniority.lower()
        job_levels = [jl.lower() for jl in item.get("job_levels", [])]
        if any(seniority_lower in jl or jl in seniority_lower for jl in job_levels):
            score += 0.08

    # Technical skills match
    if constraints.technical_skills:
        skills_lower = [s.lower() for s in item.get("skills", [])]
        name_lower = item.get("name", "").lower()
        desc_lower = item.get("description", "").lower()
        for ts in constraints.technical_skills:
            tsl = ts.lower()
            if tsl in name_lower or tsl in desc_lower or any(tsl in s for s in skills_lower):
                score += 0.10

    return score


def retrieve(
    query: str,
    constraints: Constraints | None = None,
    top_k: int | None = None,
) -> list[CatalogItem]:
    """
    Main retrieval function.

    1. Embed query.
    2. FAISS k-NN search (over-retrieve for reranking).
    3. Metadata-aware reranking.
    4. Return top_k CatalogItem objects.

    All returned items come from the catalog — no hallucination possible.
    """
    _load_resources()

    if top_k is None:
        top_k = settings.top_k_final

    k_retrieve = min(settings.top_k_retrieval, len(_catalog))

    query_vec: np.ndarray = _model.encode(  # type: ignore[union-attr]
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores, indices = _index.search(query_vec, k_retrieve)  # type: ignore[union-attr]
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    candidates: list[tuple[float, dict]] = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(_catalog):
            continue
        item = _catalog[idx]
        final_score = _rerank_score(item, constraints or Constraints(), score)
        candidates.append((final_score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:top_k]

    results: list[CatalogItem] = []
    for _, item_dict in top:
        try:
            results.append(CatalogItem(**item_dict))
        except Exception as exc:
            log.warning("Skipping malformed catalog item: %s", exc)

    return results


def get_by_names(names: list[str]) -> list[CatalogItem]:
    """
    Retrieve specific assessments by name for comparison queries.
    Fuzzy: case-insensitive substring match.
    """
    _load_resources()
    results: list[CatalogItem] = []
    for name in names:
        name_lower = name.strip().lower()
        for item in _catalog:
            if name_lower in item.get("name", "").lower():
                try:
                    results.append(CatalogItem(**item))
                except Exception:
                    pass
                break
    return results