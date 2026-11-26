"""Save per-example predictions/probabilities of an adapter on all held-out test sets
(used for bootstrap CIs, calibration, selective-prediction and error analysis)."""
import argparse, json
from pathlib import Path
import claim_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="test", choices=["test", "val"])
a = ap.parse_args()
sets = C.load_eval_sets(a.split)
model, tok = C.load_model(None if a.adapter.lower() == "none" else a.adapter)
out = {}
for name, rows in sets.items():
    preds, probs = C.predict(model, tok, rows, batch_size=16)
    out[name] = [{"gold": r["label"], "pred": p, "probs": pr, "src": r["src"]} for r, p, pr in zip(rows, preds, probs)]
    m = C.metrics(preds, [r["label"] for r in rows])
    print(a.tag, name, m["accuracy"], m["macro_f1"], flush=True)
C.RESULTS_DIR.mkdir(exist_ok=True)
C.predictions_path(a.tag, a.split).write_text(json.dumps(out), encoding="utf-8")
