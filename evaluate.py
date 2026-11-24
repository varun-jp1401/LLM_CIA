"""
Evaluation module: compare three pipeline configurations on PubMedQA.

Configurations:
  1. baseline  - Qwen3-1.7B generation only (no retrieval, no verification)
  2. retrieval - Qwen3-1.7B + hybrid retrieval (no claim verification)
  3. full      - Qwen3-1.7B + retrieval + NLI verification + repair/abstention

Metrics reported per configuration:
  - Accuracy vs. PubMedQA gold labels (yes / no / maybe)
  - Abstention rate
  - Average NLI entailment score (full pipeline only)
  - Average retrieval score (retrieval / full)
  - Runtime (seconds / sample)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

DATA_PUBMEDQA = Path(__file__).parent / "data" / "PubMedQA"


# ---------------------------------------------------------------------------
# PubMedQA loader
# ---------------------------------------------------------------------------

def load_pubmedqa_eval(max_samples: int = 500) -> list[dict]:
    """Load labeled PubMedQA examples for evaluation."""
    samples: list[dict] = []
    for fname in ["ori_pqal.json", "test_ground_truth.json"]:
        p = DATA_PUBMEDQA / "data" / fname
        if not p.exists():
            p = DATA_PUBMEDQA / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items = data.items() if isinstance(data, dict) else enumerate(data)
        for uid, item in items:
            if not isinstance(item, dict):
                continue
            q = item.get("QUESTION", item.get("question", ""))
            label = str(item.get("final_decision", item.get("answer", ""))).strip().lower()
            contexts = item.get("CONTEXTS", item.get("context", []))
            if isinstance(contexts, list):
                context_text = " ".join(str(c) for c in contexts)
            else:
                context_text = str(contexts)
            if q and label in {"yes", "no", "maybe"}:
                samples.append({
                    "id": str(uid),
                    "question": q,
                    "gold_label": label,
                    "context": context_text[:800],
                })
            if len(samples) >= max_samples:
                break
        if samples:
            break

    if not samples:
        # Synthetic fallback when PubMedQA data is not downloaded
        print("[evaluate] PubMedQA data not found — using 10 synthetic examples for demo.")
        samples = _synthetic_examples()
    return samples


def _synthetic_examples() -> list[dict]:
    """Minimal synthetic examples so evaluation can run end-to-end without real data."""
    return [
        {"id": "syn-1", "question": "Does aspirin reduce the risk of colorectal cancer?", "gold_label": "yes",
         "context": "Several randomized trials and meta-analyses indicate that regular aspirin use is associated with a reduction in colorectal cancer incidence and mortality."},
        {"id": "syn-2", "question": "Is hypertension a risk factor for stroke?", "gold_label": "yes",
         "context": "High blood pressure is the leading modifiable risk factor for ischemic and hemorrhagic stroke worldwide."},
        {"id": "syn-3", "question": "Can type 2 diabetes be cured by diet alone?", "gold_label": "no",
         "context": "While lifestyle changes including diet and exercise can achieve remission in some patients, type 2 diabetes is considered a chronic condition and diet alone does not constitute a cure for most patients."},
        {"id": "syn-4", "question": "Does metformin cause lactic acidosis in all patients?", "gold_label": "no",
         "context": "Lactic acidosis is a rare but serious complication of metformin, occurring primarily in patients with renal impairment; it does not occur in all patients."},
        {"id": "syn-5", "question": "Is there a vaccine for Alzheimer disease?", "gold_label": "no",
         "context": "As of the latest available evidence, there is no approved vaccine for Alzheimer disease, though several immunotherapy candidates are in clinical trials."},
        {"id": "syn-6", "question": "Does sleep deprivation affect cognitive function?", "gold_label": "yes",
         "context": "Extensive evidence shows that insufficient sleep impairs attention, working memory, executive function, and decision-making."},
        {"id": "syn-7", "question": "Can omega-3 fatty acids prevent cardiovascular disease?", "gold_label": "maybe",
         "context": "Evidence is mixed: some trials show modest cardiovascular benefit with high-dose omega-3 supplements, while others show no significant reduction in major adverse cardiac events."},
        {"id": "syn-8", "question": "Is depression more common in women than in men?", "gold_label": "yes",
         "context": "Epidemiological data consistently show that the prevalence of major depressive disorder is approximately twice as high in women as in men."},
        {"id": "syn-9", "question": "Does smoking cessation immediately normalize cardiovascular risk?", "gold_label": "no",
         "context": "Cardiovascular risk decreases substantially after quitting smoking but does not return to never-smoker levels immediately; full normalization may take many years."},
        {"id": "syn-10", "question": "Is antibiotic resistance a growing global health threat?", "gold_label": "yes",
         "context": "The WHO has declared antimicrobial resistance one of the top global public health threats, driven by misuse and overuse of antibiotics across human and veterinary medicine."},
    ]


# ---------------------------------------------------------------------------
# Label extraction from free-text answer
# ---------------------------------------------------------------------------

_YES_PATTERNS = re.compile(r"\b(yes|true|correct|confirmed|effective|beneficial|protective|reduces|improves|increases risk)\b", re.I)
_NO_PATTERNS = re.compile(r"\b(no\b|not\b|false|incorrect|ineffective|does not|cannot|never|unlikely)\b", re.I)
_MAYBE_PATTERNS = re.compile(r"\b(maybe|uncertain|unclear|limited|mixed|inconclusive|might|could|possible|potential)\b", re.I)


def extract_label(text: str) -> str:
    """Heuristically map a free-text answer to yes / no / maybe."""
    if not text or "abstain" in text.lower():
        return "abstain"
    t = text.lower()
    yes = len(_YES_PATTERNS.findall(t))
    no = len(_NO_PATTERNS.findall(t))
    maybe = len(_MAYBE_PATTERNS.findall(t))
    if maybe >= max(yes, no):
        return "maybe"
    if yes > no:
        return "yes"
    if no > yes:
        return "no"
    return "maybe"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _accuracy(predictions: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(p == g for p, g in zip(predictions, gold)) / len(gold)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, pipeline, retriever=None, max_samples: int = 50):
        self.pipeline = pipeline
        self.retriever = retriever
        self.max_samples = max_samples
        self.samples = load_pubmedqa_eval(max_samples)[:max_samples]

    # -- Configuration 1: baseline Qwen only --
    def run_baseline(self) -> dict:
        preds, golds, times = [], [], []
        for s in self.samples:
            t0 = time.time()
            answer = self.pipeline.generate(
                f"Answer this biomedical yes/no/maybe question with one word (yes, no, or maybe) followed by a brief explanation.\nQuestion: {s['question']}\nAnswer:"
            )
            times.append(time.time() - t0)
            preds.append(extract_label(answer))
            golds.append(s["gold_label"])
        return {
            "config": "baseline",
            "n": len(self.samples),
            "accuracy": round(_accuracy(preds, golds), 4),
            "abstention_rate": round(preds.count("abstain") / max(1, len(preds)), 4),
            "avg_time_s": round(sum(times) / max(1, len(times)), 2),
            "per_class": _per_class(preds, golds),
        }

    # -- Configuration 2: retrieval only --
    def run_retrieval(self) -> dict:
        preds, golds, times, retrieval_scores = [], [], [], []
        retriever = self.retriever or getattr(self.pipeline, "_bm25_retriever", None)
        for s in self.samples:
            t0 = time.time()
            if retriever:
                evidence = retriever.retrieve(s["question"], k=3)
                retrieval_scores.append(evidence[0]["score"] if evidence else 0.0)
                ctx = "\n\n".join(f"[{i+1}] {e['text'][:400]}" for i, e in enumerate(evidence))
            else:
                evidence = self.pipeline.retrieve(s["question"], k=3)
                retrieval_scores.append(evidence[0]["score"] if evidence else 0.0)
                ctx = "\n\n".join(f"[{i+1}] {e['text'][:400]}" for i, e in enumerate(evidence))
            answer = self.pipeline.generate(
                f"Using the evidence below, answer yes, no, or maybe and briefly explain.\nQuestion: {s['question']}\nEvidence:\n{ctx}\nAnswer:"
            )
            times.append(time.time() - t0)
            preds.append(extract_label(answer))
            golds.append(s["gold_label"])
        return {
            "config": "retrieval",
            "n": len(self.samples),
            "accuracy": round(_accuracy(preds, golds), 4),
            "abstention_rate": round(preds.count("abstain") / max(1, len(preds)), 4),
            "avg_retrieval_score": round(sum(retrieval_scores) / max(1, len(retrieval_scores)), 4),
            "avg_time_s": round(sum(times) / max(1, len(times)), 2),
            "per_class": _per_class(preds, golds),
        }

    # -- Configuration 3: full pipeline --
    def run_full(self) -> dict:
        preds, golds, times, entail_scores, abstains = [], [], [], [], []
        for s in self.samples:
            t0 = time.time()
            result = self.pipeline.run(s["question"])
            times.append(time.time() - t0)
            preds.append("abstain" if result["abstained"] else extract_label(result["answer"]))
            golds.append(s["gold_label"])
            abstains.append(result["abstained"])
            if result["claims"]:
                entail_scores.append(
                    sum(c["confidence"] for c in result["claims"]) / len(result["claims"])
                )
        return {
            "config": "full",
            "n": len(self.samples),
            "accuracy": round(_accuracy(preds, golds), 4),
            "abstention_rate": round(sum(abstains) / max(1, len(abstains)), 4),
            "avg_entailment_score": round(sum(entail_scores) / max(1, len(entail_scores)), 4) if entail_scores else 0.0,
            "avg_time_s": round(sum(times) / max(1, len(times)), 2),
            "per_class": _per_class(preds, golds),
        }


def _per_class(preds: list[str], golds: list[str]) -> dict:
    classes = {"yes", "no", "maybe"}
    out = {}
    for cls in classes:
        tp = sum(p == cls and g == cls for p, g in zip(preds, golds))
        fp = sum(p == cls and g != cls for p, g in zip(preds, golds))
        fn = sum(p != cls and g == cls for p, g in zip(preds, golds))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    return out


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Evaluate pipeline configurations on PubMedQA")
    parser.add_argument("--configs", nargs="+", choices=["baseline", "retrieval", "full"],
                        default=["baseline", "retrieval", "full"])
    parser.add_argument("--n", type=int, default=20, help="Number of evaluation samples (default 20)")
    parser.add_argument("--out", default="results/eval_results.json", help="Output JSON path")
    parser.add_argument("--use-hybrid", action="store_true", help="Use HybridRetriever instead of BM25 only")
    args = parser.parse_args()

    print("[evaluate] Loading pipeline …")
    from pipeline import MedicalPipeline
    pipe = MedicalPipeline()

    retriever = None
    if args.use_hybrid:
        try:
            from retrieval import HybridRetriever
            retriever = HybridRetriever(use_dense=True)
            print("[evaluate] HybridRetriever loaded.")
        except Exception as e:
            print(f"[evaluate] HybridRetriever unavailable: {e} — using BM25.")

    evaluator = Evaluator(pipe, retriever=retriever, max_samples=args.n)
    all_results = {}

    for cfg in args.configs:
        print(f"\n[evaluate] Running configuration: {cfg} …")
        if cfg == "baseline":
            r = evaluator.run_baseline()
        elif cfg == "retrieval":
            r = evaluator.run_retrieval()
        else:
            r = evaluator.run_full()
        all_results[cfg] = r
        print(json.dumps(r, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[evaluate] Results saved to {out_path}")
