"""Validation-based recalibration (temperature scaling) and operating-point selection for the
binary supported / unsupported decision. Parameters are fitted on validation data only and
applied unchanged to the held-out test sets."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).parent / "results"
L = ["entailment", "neutral", "contradiction"]
PAIRS = {"medquad_test_iid": "medquad_val", "medquad_test_ood": "medquad_val", "healthver_test": "healthver_dev"}
GRID = np.linspace(0.02, 0.99, 98)


def load(name):
    p = R / f"predictions_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def arrs(rows):
    g = np.array([L.index(r["gold"]) for r in rows]); p = np.clip(np.array([r["probs"] for r in rows]), 1e-9, 1)
    return g, p


def temper(p, T):
    z = np.log(p) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(g, p):
    return float(-np.mean(np.log(p[np.arange(len(g)), g])))


def ece(g, p, bins=10):
    conf = p.max(1); ok = (p.argmax(1) == g).astype(float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b) if a > 0 else (conf >= a) & (conf <= b)
        if m.sum():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def detect(g, pE, tau):
    unsup = g != 0; flagged = pE < tau; tp = np.sum(flagged & unsup)
    return {"tau": float(tau), "flag_recall": float(tp / max(1, unsup.sum())), "flag_precision": float(tp / max(1, flagged.sum())),
            "false_flag_rate": float(np.mean(flagged[~unsup])), "contra_accepted": float(np.mean(~flagged[g == 2]))}


out = {}
for tag in ("mixed", "nli_deberta", "deberta_ft"):
    test, val = load(tag), load(f"{tag}_val")
    if test is None or val is None:
        continue
    out[tag] = {}
    for ts, vs in PAIRS.items():
        gv, pv = arrs(val[vs]); gt, pt = arrs(test[ts])
        Ts = np.linspace(0.3, 6, 115)
        T = float(Ts[int(np.argmin([nll(gv, temper(pv, t)) for t in Ts]))])
        ptT = temper(pt, T)
        # operating points chosen on validation data (temperature-scaled scores)
        pvT = temper(pv, T)[:, 0]
        f1s = []
        for t in GRID:
            d = detect(gv, pvT, t); p_, r_ = d["flag_precision"], d["flag_recall"]
            f1s.append(2 * p_ * r_ / max(1e-9, p_ + r_))
        t_f1 = float(GRID[int(np.argmax(f1s))])
        ok = [t for t in GRID if detect(gv, pvT, t)["contra_accepted"] <= 0.05]
        t_safe = float(min(ok)) if ok else float(GRID[-1])
        out[tag][ts] = {"T": T, "ece_before": ece(gt, pt), "ece_after": ece(gt, ptT), "nll_before": nll(gt, pt), "nll_after": nll(gt, ptT),
                        "fixed_0.70": detect(gt, pt[:, 0], 0.70), "f1_opt": detect(gt, ptT[:, 0], t_f1), "safety_5pct": detect(gt, ptT[:, 0], t_safe),
                        "val_set": vs}
(R / "analysis_calib.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for tag, v in out.items():
    for s, d in v.items():
        print(f"{tag:12s}{s:18s} T={d['T']:.2f} ECE {d['ece_before']:.3f}->{d['ece_after']:.3f} NLL {d['nll_before']:.3f}->{d['nll_after']:.3f}")
        for k in ("fixed_0.70", "f1_opt", "safety_5pct"):
            x = d[k]
            print(f"     {k:12s} tau {x['tau']:.2f} R {x['flag_recall']:.3f} P {x['flag_precision']:.3f} FF {x['false_flag_rate']:.3f} Cacc {x['contra_accepted']:.3f}")
