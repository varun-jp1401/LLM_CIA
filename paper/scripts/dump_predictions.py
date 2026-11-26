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
if a.split == "val":
    sets = {"medquad_val": C.load_synthetic("val"), "healthver_dev": C.load_healthver("dev")}
else:
    sets = {"medquad_test_iid": C.load_synthetic("test_iid"), "medquad_test_ood": C.load_synthetic("test_ood"),
            "healthver_test": C.load_healthver("test")}
raw = {n: json.loads((C.SYN_DIR / f"{f}.json").read_text(encoding="utf-8")) for n, f in
       (("medquad_test_iid", "test_iid"), ("medquad_test_ood", "test_ood"))}
model, tok = C.load_model(None if a.adapter.lower() == "none" else a.adapter)
out = {}
for name, rows in sets.items():
    preds, probs = C.predict(model, tok, rows, batch_size=16)
    out[name] = [{"gold": r["label"], "pred": p, "probs": pr, "src": r["src"]} for r, p, pr in zip(rows, preds, probs)]
    m = C.metrics(preds, [r["label"] for r in rows])
    print(a.tag, name, m["accuracy"], m["macro_f1"], flush=True)
Path(C.ROOT / "results").mkdir(exist_ok=True)
(C.ROOT / "results" / (f"predictions_{a.tag}.json" if a.split == "test" else f"predictions_{a.tag}_val.json")).write_text(json.dumps(out), encoding="utf-8")
