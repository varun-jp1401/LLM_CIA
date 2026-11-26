"""
Evaluate a claim-verification adapter (and optionally the untouched base model).

Reports precision / recall / F1 per class on:
  * medquad_test_iid  - held-out TOPICS, same edit families as training
  * medquad_test_ood  - held-out topics AND unseen edit families / unsupported-addition templates
  * healthver_test    - REAL claims (human-annotated), the honest benchmark
  * train_probe       - 600 training rows; train-vs-heldout gap is the memorisation check

    python evaluate_claims.py --adapter models/qwen3-1.7b-lora-claimverify --compare-base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import claim_common as C


def run(model, tok, label: str, sets: dict) -> dict:
    res = {}
    for name, rows in sets.items():
        preds, _ = C.predict(model, tok, rows, batch_size=16)
        m = C.metrics(preds, [r["label"] for r in rows])
        res[name] = m
        pc = m["per_class"]
        print(f"[{label}] {name:18s} n={m['n']:4d} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"P={m['macro_precision']:.3f} R={m['macro_recall']:.3f} | "
              + " ".join(f"{k[:4]} P/R {v['precision']:.2f}/{v['recall']:.2f}" for k, v in pc.items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(C.ROOT / "models" / "qwen3-1.7b-lora-claimverify"))
    ap.add_argument("--compare-base", action="store_true")
    ap.add_argument("--out", default=str(C.ROOT / "results" / "eval_claim_verification.json"))
    args = ap.parse_args()

    sets = {
        "medquad_test_iid": C.load_synthetic("test_iid"),
        "medquad_test_ood": C.load_synthetic("test_ood"),
        "healthver_test": C.load_healthver("test"),
        "train_probe": C.sample(C.load_healthver("train"), 300, 7) + C.sample(C.load_synthetic("train"), 300, 7),
    }
    results = {"_meta": {"adapter": args.adapter, "base": C.BASE_MODEL}}

    model, tok = C.load_model(args.adapter)
    results["adapter"] = run(model, tok, "adapter", sets)
    if args.compare_base:
        base, tok = C.load_model(None)
        results["base_zero_shot"] = run(base, tok, "base", {k: v for k, v in sets.items() if k != "train_probe"})
        del base
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
