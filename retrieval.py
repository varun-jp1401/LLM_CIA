"""
Hybrid retrieval module: BM25 lexical + PubMedBERT dense (FAISS) with RRF fusion.

Falls back gracefully if FAISS or sentence-transformers are unavailable —
the app then uses pure BM25 from pipeline.py.
"""
from __future__ import annotations

import json
import math
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

DATA_MEDQUAD = Path(__file__).parent / "data" / "MedQuAD"
DATA_PUBMEDQA = Path(__file__).parent / "data" / "PubMedQA"
INDEX_CACHE = Path(__file__).parent / "data" / "faiss_index.pkl"

# PubMedBERT sentence-transformer (small, downloads ~440 MB once)
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 4]


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


# ---------------------------------------------------------------------------
# MedQuAD loader
# ---------------------------------------------------------------------------

def load_medquad() -> list[dict]:
    docs: list[dict] = []
    if not DATA_MEDQUAD.exists():
        return docs
    for p in DATA_MEDQUAD.rglob("*.xml"):
        try:
            root = ET.parse(p).getroot()
            extract = lambda node: " ".join(t.strip() for t in node.itertext() if t.strip())
            pairs = root.findall(".//QAPair") or [root]
            for pair in pairs:
                q = " ".join(extract(x) for x in pair.findall(".//Question"))
                a = " ".join(extract(x) for x in pair.findall(".//Answer"))
                if q and len(a) >= 40:
                    docs.append({
                        "text": f"Question: {q}\nAnswer: {a}",
                        "source": str(p.relative_to(DATA_MEDQUAD)),
                        "corpus": "MedQuAD",
                    })
        except (ET.ParseError, OSError):
            continue
    return docs


def load_pubmedqa() -> list[dict]:
    """Load PubMedQA labeled split as retrieval passages."""
    docs: list[dict] = []
    for fname in ["ori_pqal.json", "test_ground_truth.json"]:
        p = DATA_PUBMEDQA / "data" / fname
        if not p.exists():
            # Try directly under DATA_PUBMEDQA
            p = DATA_PUBMEDQA / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            items = data.items()
        else:
            items = enumerate(data)
        for uid, item in items:
            if isinstance(item, dict):
                q = item.get("QUESTION", item.get("question", ""))
                # Long abstract contexts
                contexts = item.get("CONTEXTS", item.get("context", []))
                if isinstance(contexts, list):
                    context_text = " ".join(str(c) for c in contexts)
                else:
                    context_text = str(contexts)
                ans = item.get("final_decision", item.get("answer", ""))
                if q and context_text:
                    docs.append({
                        "text": f"Question: {q}\nAnswer: {context_text[:800]}",
                        "source": f"PubMedQA/{fname}/{uid}",
                        "corpus": "PubMedQA",
                        "label": str(ans),
                    })
    return docs


# ---------------------------------------------------------------------------
# BM25 scorer (matches pipeline.py but extracted for reuse)
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        # Pre-compute doc term sets for speed
        self._doc_terms = [_terms(d["text"]) for d in docs]
        self._n = max(1, len(docs))

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        q_terms = _terms(query)
        if not q_terms or not self.docs:
            return []
        dfs = {t: sum(1 for dt in self._doc_terms if t in dt) for t in q_terms}
        ranked = []
        for doc, dt in zip(self.docs, self._doc_terms):
            score = sum(
                math.log(1 + (self._n - dfs[t] + 0.5) / (dfs[t] + 0.5))
                for t in q_terms if t in dt
            )
            score /= 1 - 0.75 + 0.75 * len(dt) / 120
            # Question-match bonus
            src_q = doc["text"].split("\nAnswer:", 1)[0].lower()
            src_q_terms = _terms(src_q)
            score += 4.0 * len(q_terms & src_q_terms) / max(1, len(q_terms))
            entity_terms = q_terms - {
                "what", "are", "is", "the", "common", "can", "how", "does",
                "do", "help", "reduce", "risk", "of", "for", "and", "or", "with",
            }
            score += 18.0 * len(entity_terms & src_q_terms) / max(1, len(entity_terms))
            if score:
                ranked.append((score, doc))
        return [
            {**d, "score": round(s, 3), "retriever": "bm25"}
            for s, d in sorted(ranked, reverse=True, key=lambda x: x[0])[:k]
        ]


# ---------------------------------------------------------------------------
# Dense retriever with FAISS
# ---------------------------------------------------------------------------

class DenseRetriever:
    """PubMedBERT sentence-transformer + FAISS flat-IP index."""

    def __init__(self, docs: list[dict], model_name: str = EMBED_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install sentence-transformers and faiss-cpu to use dense retrieval.") from exc

        self.docs = docs
        self._model_name = model_name
        self._embedder: Optional[object] = None
        self._index: Optional[object] = None
        self._dim: int = 0

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        if self._embedder is None:
            try:
                self._embedder = SentenceTransformer(self._model_name)
            except Exception:
                self._embedder = SentenceTransformer(FALLBACK_EMBED_MODEL)

    def build_index(self, texts: list[str] | None = None, batch_size: int = 64):
        import faiss
        self._load_model()
        if texts is None:
            texts = [d["text"][:512] for d in self.docs]
        embs = self._embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self._dim = embs.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embs)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if self._index is None:
            raise RuntimeError("Call build_index() first.")
        self._load_model()
        qemb = self._embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, idxs = self._index.search(qemb, k)
        results = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            results.append({**self.docs[i], "score": round(float(s), 4), "retriever": "dense"})
        return results

    def save(self, path: Path = INDEX_CACHE):
        import faiss
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"docs": self.docs, "index": faiss.serialize_index(self._index), "dim": self._dim, "model": self._model_name},
                f,
            )

    @classmethod
    def load(cls, path: Path = INDEX_CACHE) -> "DenseRetriever":
        import faiss
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["docs"], state["model"])
        obj._index = faiss.deserialize_index(state["index"])
        obj._dim = state["dim"]
        return obj


# ---------------------------------------------------------------------------
# Hybrid RRF fusion
# ---------------------------------------------------------------------------

def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def hybrid_retrieve(
    query: str,
    bm25: BM25Retriever,
    dense: Optional[DenseRetriever],
    k: int = 5,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[dict]:
    """Reciprocal-rank fusion of BM25 and dense results."""
    bm25_results = bm25.retrieve(query, k=k * 3)

    if dense is None:
        return bm25_results[:k]

    try:
        dense_results = dense.retrieve(query, k=k * 3)
    except Exception:
        return bm25_results[:k]

    # Build a score map keyed by source identifier
    scores: dict[str, dict] = {}

    for rank, doc in enumerate(bm25_results, 1):
        key = doc["source"]
        if key not in scores:
            scores[key] = {**doc, "rrf_score": 0.0}
        scores[key]["rrf_score"] += bm25_weight * _rrf_score(rank)

    for rank, doc in enumerate(dense_results, 1):
        key = doc["source"]
        if key not in scores:
            scores[key] = {**doc, "rrf_score": 0.0}
        scores[key]["rrf_score"] += dense_weight * _rrf_score(rank)

    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    for item in ranked:
        item["score"] = round(item["rrf_score"], 4)
        item["retriever"] = "hybrid_rrf"
    return ranked[:k]


# ---------------------------------------------------------------------------
# Top-level convenience: build or load the full hybrid index
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combines BM25 + optional FAISS dense retrieval across MedQuAD + PubMedQA."""

    def __init__(self, use_dense: bool = True, rebuild: bool = False):
        all_docs = load_medquad() + load_pubmedqa()
        self.bm25 = BM25Retriever(all_docs)
        self.dense: Optional[DenseRetriever] = None

        if use_dense:
            try:
                if not rebuild and INDEX_CACHE.exists():
                    print("[retrieval] Loading cached FAISS index …")
                    self.dense = DenseRetriever.load(INDEX_CACHE)
                else:
                    if all_docs:
                        print(f"[retrieval] Building FAISS index over {len(all_docs)} docs …")
                        self.dense = DenseRetriever(all_docs)
                        self.dense.build_index()
                        self.dense.save(INDEX_CACHE)
                    else:
                        print("[retrieval] No documents found; dense index skipped.")
            except ImportError as e:
                print(f"[retrieval] Dense retrieval unavailable: {e}")
            except Exception as e:
                print(f"[retrieval] FAISS index build failed: {e}")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        return hybrid_retrieve(query, self.bm25, self.dense, k=k)
