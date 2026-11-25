"""
Build (or rebuild) the FAISS dense retrieval index over MedQuAD + PubMedQA.

This is a one-time step that caches the index to data/faiss_index.pkl.
Subsequent runs of the app / evaluator load the cached index automatically.

Usage:
    python build_index.py                         # build with default PubMedBERT model
    python build_index.py --max-docs 10000        # limit doc count (faster)
    python build_index.py --model fast            # use all-MiniLM-L6-v2 (much faster)
    python build_index.py --rebuild               # force rebuild even if cache exists
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

INDEX_CACHE = Path(__file__).parent / "data" / "faiss_index.pkl"

EMBED_MODELS = {
    "pubmedbert": "pritamdeka/S-PubMedBert-MS-MARCO",   # domain-specific, recommended
    "fast":       "sentence-transformers/all-MiniLM-L6-v2",  # generic but very fast
    "minilm":     "sentence-transformers/all-MiniLM-L6-v2",
}


def main():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Maximum number of documents to index (0 = all)")
    parser.add_argument("--model", choices=list(EMBED_MODELS.keys()), default="pubmedbert",
                        help="Embedding model to use")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild even if cache already exists")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Encoding batch size (increase for GPU)")
    args = parser.parse_args()

    # Check dependencies
    try:
        import faiss  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as e:
        print(f"[build_index] Missing dependency: {e}")
        print("Install with: pip install faiss-cpu sentence-transformers")
        raise SystemExit(1)

    if INDEX_CACHE.exists() and not args.rebuild:
        print(f"[build_index] Cache already exists at {INDEX_CACHE}")
        print("  Use --rebuild to force a rebuild.")
        return

    from retrieval import load_medquad, load_pubmedqa, DenseRetriever

    print("[build_index] Loading documents ...")
    t0 = time.time()
    docs = load_medquad() + load_pubmedqa()
    print(f"[build_index] Loaded {len(docs)} documents in {time.time()-t0:.1f}s")

    if args.max_docs and args.max_docs < len(docs):
        import random
        random.seed(0)
        random.shuffle(docs)
        docs = docs[: args.max_docs]
        print(f"[build_index] Limiting to {len(docs)} documents (--max-docs)")

    model_name = EMBED_MODELS[args.model]
    print(f"[build_index] Embedding model: {model_name}")
    print(f"[build_index] Building FAISS index over {len(docs)} docs "
          f"(batch_size={args.batch_size}) …")

    retriever = DenseRetriever(docs, model_name=model_name)
    t1 = time.time()
    retriever.build_index(batch_size=args.batch_size)
    elapsed = time.time() - t1
    print(f"[build_index] Index built in {elapsed:.1f}s")

    retriever.save(INDEX_CACHE)
    print(f"[build_index] Saved to {INDEX_CACHE}  ({INDEX_CACHE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
