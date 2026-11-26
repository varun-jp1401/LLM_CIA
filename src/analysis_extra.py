"""Extra analyses from dumped predictions: unsupported-claim detection at the deployed threshold,
average precision, breakdown by MedQuAD source collection and by claim / evidence length."""
import json
from pathlib import Path
import numpy as np
from analysis_common import R, L, SETS, load
from sklearn.metrics import average_precision_score
import claim_common as C

TAU = 0.70


meta = {}
for n, f in (("medquad_test_iid", "test_iid"), ("medquad_test_ood", "test_ood")):
    raw = json.loads((C.SYN_DIR / f"{f}.json").read_text(encoding="utf-8"))
    meta[n] = [{"source": r["source"], "claim_w": len(r["claim"].split()), "evid_w": len(r["evidence"].split())} for r in raw]
hv = C.load_healthver("test")
meta["healthver_test"] = [{"source": "HealthVer", "claim_w": len(r["claim"].split()), "evid_w": len(r["evidence"].split())} for r in hv]

out = {"tau": TAU, "detection": {}, "breakdown": {}}
tags = ["mixed", "nli_deberta", "base", "real_only", "synth_only", "fewshot", "deberta_ft"]
for t in tags:
    P = load(t)
    if P is None:
        continue
    out["detection"][t] = {}
    for s in SETS:
        g = np.array([r["gold"] for r in P[s]]); pE = np.array([r["probs"][0] for r in P[s]])
        unsup = g != "entailment"; flagged = pE < TAU
        tp = np.sum(flagged & unsup)
        d = {"flag_recall": float(tp / max(1, unsup.sum())), "flag_precision": float(tp / max(1, flagged.sum())),
             "false_flag_rate": float(np.mean(flagged[~unsup])),
             "contra_accepted": float(np.mean(~flagged[g == "contradiction"])),
             "ap_unsupported": float(average_precision_score(unsup, 1 - pE))}
        out["detection"][t][s] = d

P = load("mixed")
for s in SETS:
    pred_ok = np.array([r["gold"] == r["pred"] for r in P[s]])
    m = meta[s]
    srcs = {}
    for ok, mi in zip(pred_ok, m):
        srcs.setdefault(mi["source"], []).append(ok)
    out["breakdown"][s] = {"source": {k: [float(np.mean(v)), len(v)] for k, v in srcs.items()}}
    for key in ("claim_w", "evid_w"):
        vals = np.array([mi[key] for mi in m]); q = np.quantile(vals, [1 / 3, 2 / 3])
        bins = [(vals <= q[0]), (vals > q[0]) & (vals <= q[1]), (vals > q[1])]
        out["breakdown"][s][key] = [{"range": [int(vals[b].min()), int(vals[b].max())], "acc": float(pred_ok[b].mean()), "n": int(b.sum())} for b in bins]
(R / "analysis_extra.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for t, v in out["detection"].items():
    for s, d in v.items():
        print(f"{t:12s}{s:18s} flagR {d['flag_recall']:.3f} flagP {d['flag_precision']:.3f} falseflag {d['false_flag_rate']:.3f} Caccepted {d['contra_accepted']:.3f} AP {d['ap_unsupported']:.3f}")
print(json.dumps(out["breakdown"]["healthver_test"], indent=0)[:900])
print(out["breakdown"]["medquad_test_iid"]["source"])
