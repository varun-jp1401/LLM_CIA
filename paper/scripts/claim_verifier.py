"""Claim-level verification for the demo: evidence retrieval -> LoRA-adapted Qwen3-1.7B verifier
(entailment / neutral / contradiction), shown next to the zero-shot DeBERTa NLI baseline.

The per-window classifier is the model evaluated in the paper. The aggregation of several windows
into one verdict (see `aggregate`) is a demo heuristic that was NOT evaluated."""
from __future__ import annotations

import re

import claim_common as C

CONF_MIN = 0.50      # a window must be at least this confident to count as entailing / contradicting
MIN_OVERLAP = 0.35   # ...and share at least this fraction of the claim's words (guards against off-topic windows)


_STOP = {"the", "and", "are", "can", "may", "for", "with", "that", "this", "from", "has", "have", "been", "not", "but", "all", "any",
         "will", "also", "such", "than", "then", "into", "more", "most", "some", "many", "often", "when", "which", "taking", "used",
         "use", "who", "was", "were", "its", "their", "they", "them", "you", "your", "does", "cause", "causes", "lead", "help", "helps"}


def _terms(text: str) -> set[str]:
    """Content words only (stop-words removed) so overlap reflects topical match, not function words."""
    return {t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in _STOP}


def evidence_windows(claim: str, passages: list[dict], n: int = 4, window: int = 2) -> list[dict]:
    """Rank windows of `window` consecutive answer sentences by lexical overlap with the claim.
    Windows (~40-60 words) match the evidence granularity the verifier was trained on."""
    ct = _terms(claim)
    cands = []
    for p in passages:
        answer = p["text"].split("\nAnswer:", 1)[-1]
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", answer) if len(x.split()) >= 4]
        for i in range(max(1, len(sents) - window + 1)):
            text = " ".join(sents[i:i + window])
            if text:
                cands.append((len(ct & _terms(text)) / max(1, len(ct)), text, p["source"]))
    cands.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for score, text, src in cands:
        key = text[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "source": src, "overlap": round(score, 3)})
        if len(out) == n:
            break
    return out


def aggregate(windows: list[dict]) -> tuple[str, str]:
    """Demo heuristic: (verdict, explanation). Not evaluated in the paper."""
    rel = [w for w in windows if w["overlap"] >= MIN_OVERLAP]
    ent = [w for w in rel if w["pred"] == "entailment" and w["probs"]["entailment"] >= CONF_MIN]
    con = [w for w in rel if w["pred"] == "contradiction" and w["probs"]["contradiction"] >= CONF_MIN]
    if ent and con:
        return "conflicting", "Some evidence windows support the claim and others contradict it."
    if con:
        return "contradicted", f"{len(con)} of {len(rel)} relevant evidence windows contradict the claim."
    if ent:
        return "supported", f"{len(ent)} of {len(rel)} relevant evidence windows support the claim."
    if not rel:
        return "not_enough_evidence", "No retrieved evidence was relevant enough to the claim."
    return "not_enough_evidence", "No relevant evidence window clearly supports or contradicts the claim."


def check_claim(pipe, claim: str, retriever=None, k: int = 5, n_windows: int = 4) -> dict:
    passages = retriever.retrieve(claim, k=k) if retriever else pipe.retrieve(claim, k=k)
    windows = evidence_windows(claim, passages, n=n_windows)
    out = {"claim": claim, "passages": passages, "windows": windows, "verdict": "not_enough_evidence",
           "explanation": "No evidence could be retrieved for this claim.", "backend": None}
    if not windows:
        return out
    rows = [{"claim": claim, "evidence": w["text"]} for w in windows]
    if getattr(pipe, "claim_adapter_loaded", False):
        with pipe.adapter("claimverify"):
            preds, probs = C.predict(pipe.model, pipe.tokenizer, rows, batch_size=len(rows))
        out["backend"] = "Qwen3-1.7B + claim-verification LoRA"
        for w, pr, pb in zip(windows, preds, probs):
            w["pred"], w["probs"] = pr, dict(zip(C.LABELS, pb))
        out["verdict"], out["explanation"] = aggregate(windows)
    else:  # adapter missing: fall back to the DeBERTa NLI scores only
        out["backend"] = "DeBERTa NLI only (claim adapter not loaded)"
        for w in windows:
            w["pred"], w["probs"] = "neutral", {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
    # independent zero-shot NLI baseline on the same retrieved passages (the deployed pipeline's verifier)
    nli = pipe.verify(claim, passages)
    out["nli"] = {"label": nli["label"], "entailment_score": round(nli["entailment_score"], 3), "source": nli.get("source")}
    return out
