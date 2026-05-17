"""
Build a FAISS semantic index from the scraped catalog JSON.

Usage:
    python scripts/build_index.py \
        --catalog data/shl_catalog.json \
        --index   data/shl_faiss.index \
        --meta    data/shl_faiss_meta.json

The index stores embeddings of a rich text field:
    "{name} — {description} — skills: {skills}"

The meta file stores a parallel list of catalog dicts (same order as FAISS rows)
so retrieval can return full CatalogItem objects.
"""

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_text(item: dict) -> str:
    """Construct a rich text representation for embedding."""
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
    if tt:
        type_map = {
            "A": "ability cognitive aptitude",
            "B": "biodata background",
            "C": "competency behavioural",
            "E": "exercise simulation work sample",
            "J": "job focused",
            "K": "knowledge skills technical",
            "P": "personality behaviour traits",
            "S": "situational judgement simulation",
        }
        parts.append(type_map.get(tt, ""))
    return " | ".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/shl_catalog.json")
    parser.add_argument("--index", default="data/shl_faiss.index")
    parser.add_argument("--meta", default="data/shl_faiss_meta.json")
    parser.add_argument("--model", default=EMBED_MODEL)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    with open(catalog_path, encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    log.info("Loaded %d items from catalog.", len(items))

    log.info("Loading embedding model: %s", args.model)
    model = SentenceTransformer(args.model)

    texts = [build_text(item) for item in items]
    log.info("Encoding %d texts…", len(texts))
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine via inner-product
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    log.info("Embedding dim: %d", dim)

    index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vecs
    index.add(embeddings)
    log.info("FAISS index size: %d", index.ntotal)

    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, args.index)
    log.info("Saved FAISS index to %s", args.index)

    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    log.info("Saved metadata to %s", args.meta)


if __name__ == "__main__":
    main()