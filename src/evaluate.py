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
import random
import re
import time
from pathlib import Path
from typing import Optional

DATA_PUBMEDQA = Path(__file__).resolve().parent.parent / "data" / "PubMedQA"
DATA_FINETUNE_SPLITS = Path(__file__).resolve().parent.parent / "data" / "finetune_splits"


# ---------------------------------------------------------------------------
# PubMedQA loader
# ---------------------------------------------------------------------------

def load_pubmedqa_eval(max_samples: int = 500, seed: int = 42, split: str = "stratified") -> list[dict]:
    """Load labeled PubMedQA examples for evaluation."""
    max_samples = max(0, max_samples)
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
        if samples:
            break

    if not samples:
        # Synthetic fallback when PubMedQA data is not downloaded
        print("[evaluate] PubMedQA data not found — using 10 synthetic examples for demo.")
        samples = _synthetic_examples()
    if split in {"stratified", "balanced"} and samples:
        rng = random.Random(seed)
        labels = ("yes", "no", "maybe")
        groups = {label: [s for s in samples if s["gold_label"] == label] for label in labels}
        if split == "balanced":
            per_class = max_samples // len(labels)
            counts = {label: per_class for label in labels}
            for label in labels[:max_samples - sum(counts.values())]:
                counts[label] += 1
        else:
            sample_count = min(max_samples, len(samples))
            expected = {label: sample_count * len(rows) / len(samples) for label, rows in groups.items()}
            counts = {label: int(value) for label, value in expected.items()}
            for label in sorted(labels, key=lambda key: expected[key] - counts[key], reverse=True)[:sample_count - sum(counts.values())]:
                counts[label] += 1
        samples = [s for label in labels for s in rng.sample(groups[label], min(counts[label], len(groups[label])))]
        rng.shuffle(samples)
    elif split == "first":
        samples = samples[:max_samples]
    else:
        raise ValueError("split must be 'stratified', 'balanced', or 'first'")
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


def load_finetune_holdout(split: str) -> list[dict]:
    """Load the explicitly saved validation/test partition from finetune.py."""
    if split not in {"validation", "test"}:
        raise ValueError("Only validation or test can be loaded as a holdout.")
    path = DATA_FINETUNE_SPLITS / f"{split}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("task") == "pubmedqa"]
    samples = []
    for row in rows:
        input_text = row.get("input", "")
        # Saved PubMedQA rows use "Evidence: " on the same line. Accept both
        # same-line and next-line forms, and never return the whole input as
        # evidence when the marker is absent.
        marker = re.search(r"\bEvidence:\s*", input_text)
        context = input_text[marker.end():].strip() if marker else ""
        samples.append({"id": row.get("id", ""), "question": row["question"],
                        "gold_label": row["output"], "context": context})
    return samples


# ---------------------------------------------------------------------------
# Label extraction from free-text answer
# ---------------------------------------------------------------------------

_YES_PATTERNS = re.compile(r"\b(yes|true|correct|confirmed|effective|beneficial|protective|reduces|improves|increases risk)\b", re.I)
_NO_PATTERNS = re.compile(r"\b(no\b|not\b|false|incorrect|ineffective|does not|cannot|never|unlikely)\b", re.I)
_MAYBE_PATTERNS = re.compile(r"\b(maybe|uncertain|unclear|limited|mixed|inconclusive|might|could|possible|potential)\b", re.I)


def extract_label(text: str) -> str:
    """Read an explicit first-line yes/no/maybe label; otherwise mark invalid."""
    if not text or "i can’t verify" in text.lower() or "i can't verify" in text.lower():
        return "abstain"

    # Strip safety disclaimer footer so words like 'did not' don't corrupt the answer
    clean_text = text.split("Evidence check:", 1)[0].strip()

    first_line = clean_text.splitlines()[0].strip().lower() if clean_text else ""
    match = re.fullmatch(r"(?:\*\*)?(yes|no|maybe)(?:\*\*)?(?:[.!,:;\s].*)?", first_line)
    return match.group(1) if match else "invalid"


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
    def __init__(self, pipeline, retriever=None, max_samples: int = 50, seed: int = 42, split: str = "stratified"):
        self.pipeline = pipeline
        self.retriever = retriever
        self.max_samples = max_samples
        self.samples = load_pubmedqa_eval(max_samples, seed=seed, split=split)
        self.debug_outputs = []

    def _record(self, sample: dict, raw: str, prediction: str, config: str) -> None:
        self.debug_outputs.append({
            "id": sample["id"], "gold": sample["gold_label"],
            "prediction": prediction, "raw_output": raw[:500], "config": config,
        })

    # -- Configuration 1: baseline Qwen only --
    def run_baseline(self) -> dict:
        preds, golds, times = [], [], []
        total = len(self.samples)
        for i, s in enumerate(self.samples, 1):
            t0 = time.time()
            prompt = ("Answer the biomedical research question using the abstract evidence. "
                      "Output exactly one label: yes, no, or maybe.\n\n"
                      f"Question: {s['question']}\nEvidence: {s.get('context', '')}")
            if hasattr(self.pipeline, "classify_choice"):
                pred, scores = self.pipeline.classify_choice(prompt)
                answer = f"{pred} {scores}"
            else:
                answer = self.pipeline.generate(prompt)
            elapsed = time.time() - t0
            times.append(elapsed)
            pred = answer.split(" ", 1)[0] if answer.startswith(("yes ", "no ", "maybe ")) else extract_label(answer)
            self._record(s, answer, pred, "baseline")
            preds.append(pred)
            golds.append(s["gold_label"])
            print(f"  [baseline {i}/{total}] {elapsed:.2f}s | pred={pred} gold={s['gold_label']}", flush=True)
        return {
            "config": "baseline",
            "n": len(self.samples),
            "abstention_rate": round(preds.count("abstain") / max(1, len(preds)), 4),
            "avg_time_s": round(sum(times) / max(1, len(times)), 2),
            **_summary(preds, golds),
        }

    # -- Configuration 2: retrieval only --
    def run_retrieval(self) -> dict:
        preds, golds, times, retrieval_scores = [], [], [], []
        retriever = self.retriever or getattr(self.pipeline, "_bm25_retriever", None)
        total = len(self.samples)
        for i, s in enumerate(self.samples, 1):
            t0 = time.time()
            if retriever:
                evidence = retriever.retrieve(s["question"], k=3)
                retrieval_scores.append(evidence[0]["score"] if evidence else 0.0)
                ctx = "\n\n".join(f"[{j+1}] {e['text'][:400]}" for j, e in enumerate(evidence))
            else:
                evidence = self.pipeline.retrieve(s["question"], k=3)
                retrieval_scores.append(evidence[0]["score"] if evidence else 0.0)
                ctx = "\n\n".join(f"[{j+1}] {e['text'][:400]}" for j, e in enumerate(evidence))
            prompt = ("Answer the biomedical research question using the abstract evidence. "
                      "Output exactly one label: yes, no, or maybe.\n\n"
                      f"Question: {s['question']}\nEvidence: {ctx}")
            if hasattr(self.pipeline, "classify_choice"):
                pred, scores = self.pipeline.classify_choice(prompt)
                answer = f"{pred} {scores}"
            else:
                answer = self.pipeline.generate(prompt)
            elapsed = time.time() - t0
            times.append(elapsed)
            pred = answer.split(" ", 1)[0] if answer.startswith(("yes ", "no ", "maybe ")) else extract_label(answer)
            self._record(s, answer, pred, "retrieval")
            preds.append(pred)
            golds.append(s["gold_label"])
            print(f"  [retrieval {i}/{total}] {elapsed:.2f}s | pred={pred} gold={s['gold_label']}", flush=True)
        return {
            "config": "retrieval",
            "n": len(self.samples),
            "abstention_rate": round(preds.count("abstain") / max(1, len(preds)), 4),
            "avg_retrieval_score": round(sum(retrieval_scores) / max(1, len(retrieval_scores)), 4),
            "avg_time_s": round(sum(times) / max(1, len(times)), 2),
            **_summary(preds, golds),
        }

    # -- Configuration 3: full pipeline --
    def run_full(self) -> dict:
        preds, golds, times, entail_scores, abstains = [], [], [], [], []
        total = len(self.samples)
        for i, s in enumerate(self.samples, 1):
            t0 = time.time()
            result = self.pipeline.run(s["question"], retriever=self.retriever, is_qa_choice=True, extra_evidence=s.get("context", ""))
            elapsed = time.time() - t0
            times.append(elapsed)
            pred = "abstain" if result["abstained"] else (result.get("choice_label") or extract_label(result["answer"]))
            self._record(s, result.get("draft", result["answer"]), pred, "full")
            preds.append(pred)
            golds.append(s["gold_label"])
            abstains.append(result["abstained"])
            if result["claims"]:
                entail_scores.append(
                    sum(c["confidence"] for c in result["claims"]) / len(result["claims"])
                )
            print(f"  [full {i}/{total}] {elapsed:.2f}s | pred={pred} gold={s['gold_label']} abstained={result['abstained']}", flush=True)
        return {
            "config": "full",
            "n": len(self.samples),
            "abstention_rate": round(sum(abstains) / max(1, len(abstains)), 4),
            "avg_entailment_score": round(sum(entail_scores) / max(1, len(entail_scores)), 4) if entail_scores else 0.0,
            "avg_time_s": round(sum(times) / max(1, len(times)), 2),
            **_summary(preds, golds),
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


def _summary(preds: list[str], golds: list[str]) -> dict:
    classes = ("yes", "no", "maybe")
    per_class = _per_class(preds, golds)
    confusion = {gold: {pred: sum(p == pred and g == gold for p, g in zip(preds, golds))
                        for pred in (*classes, "abstain", "invalid")} for gold in classes}
    return {
        "accuracy": round(_accuracy(preds, golds), 4),
        "macro_f1": round(sum(per_class[c]["f1"] for c in classes) / 3, 4),
        "prediction_counts": {label: preds.count(label) for label in (*classes, "abstain", "invalid")},
        "per_class": per_class,
        "confusion_matrix": confusion,
        "invalid_rate": round(preds.count("invalid") / max(1, len(preds)), 4),
    }


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
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter to load into the pipeline (fine-tuned model)")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate the base model without the adapter on the same sample")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling")
    parser.add_argument("--split", choices=["stratified", "balanced", "first"], default="stratified",
                        help="Seeded proportional sample (recommended), class-balanced diagnostic, or file order")
    parser.add_argument("--debug-output", action="store_true", help="Include first 500 characters of each generated answer")
    parser.add_argument("--held-out-split", choices=["validation", "test"], default=None,
                        help="Evaluate on the saved finetune_splits partition instead of a new sample")
    args = parser.parse_args()

    print("[evaluate] Loading pipeline …")
    from pipeline import MedicalPipeline
    pipe = MedicalPipeline(adapter_path=args.adapter)
    if args.adapter:
        print(f"[evaluate] Using fine-tuned model with adapter: {args.adapter}")

    retriever = None
    if args.use_hybrid:
        try:
            from retrieval import HybridRetriever
            retriever = HybridRetriever(use_dense=True, include_pubmedqa=False)
            print("[evaluate] HybridRetriever loaded.")
        except Exception as e:
            print(f"[evaluate] HybridRetriever unavailable: {e} — using BM25.")

    evaluator = Evaluator(pipe, retriever=retriever, max_samples=args.n, seed=args.seed, split=args.split)
    if args.held_out_split:
        evaluator.samples = load_finetune_holdout(args.held_out_split)
    all_results = {"_meta": {
        "adapter": args.adapter,
        "sample_seed": args.seed,
        "sample_strategy": args.split,
        "held_out_split": args.held_out_split,
        "sample_ids": [s["id"] for s in evaluator.samples],
        "class_counts": {label: sum(s["gold_label"] == label for s in evaluator.samples) for label in ("yes", "no", "maybe")},
    }}

    def run_configs(target, prefix=""):
        for cfg in args.configs:
            print(f"\n[evaluate] Running {prefix}{cfg} ...")
            r = target.run_baseline() if cfg == "baseline" else target.run_retrieval() if cfg == "retrieval" else target.run_full()
            all_results[f"{prefix}{cfg}"] = r
            if args.debug_output:
                all_results.setdefault("_debug_outputs", []).extend(target.debug_outputs)
                target.debug_outputs.clear()
            print(json.dumps(r, indent=2))

    run_configs(evaluator, "adapter_" if args.adapter else "")
    if args.compare_base and args.adapter:
        print("\n[evaluate] Loading base model for same-sample comparison ...")
        base_pipe = MedicalPipeline(adapter_path=None)
        base_evaluator = Evaluator(base_pipe, retriever=retriever, max_samples=args.n, seed=args.seed, split=args.split)
        if args.held_out_split:
            base_evaluator.samples = load_finetune_holdout(args.held_out_split)
        run_configs(base_evaluator, "base_")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[evaluate] Results saved to {out_path}")
