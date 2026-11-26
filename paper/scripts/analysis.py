"""Statistics for the paper from dumped per-example predictions: bootstrap CIs, calibration,
selective prediction, per-edit-type accuracy, paired bootstrap differences."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).parent / "results"
L = ["entailment", "neutral", "contradiction"]
SETS = ["medquad_test_iid", "medquad_test_ood", "healthver_test"]
rng = np.random.default_rng(0)


def load(tag):
    p = R / f"predictions_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def arr(rows):
    g = np.array([L.index(r["gold"]) for r in rows]); p = np.array([L.index(r["pred"]) for r in rows])
    pr = np.array([r["probs"] for r in rows]); return g, p, pr


def macro(g, p):
    ps, rs, fs = [], [], []
    for k in range(3):
        tp = np.sum((p == k) & (g == k)); fp = np.sum((p == k) & (g != k)); fn = np.sum((p != k) & (g == k))
        pr_, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        ps.append(pr_); rs.append(rc); fs.append(2 * pr_ * rc / max(1e-9, pr_ + rc))
    return dict(acc=float(np.mean(g == p)), P=float(np.mean(ps)), R=float(np.mean(rs)), F1=float(np.mean(fs)))


def boot(g, p, B=2000):
    n = len(g); out = {k: [] for k in ("acc", "P", "R", "F1")}
    for _ in range(B):
        i = rng.integers(0, n, n); m = macro(g[i], p[i])
        for k in out: out[k].append(m[k])
    pt = macro(g, p)
    return {k: {"point": pt[k], "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))} for k, v in out.items()}


def paired_diff(g, p1, p2, B=2000):
    n = len(g); d = []
    for _ in range(B):
        i = rng.integers(0, n, n); d.append(macro(g[i], p1[i])["F1"] - macro(g[i], p2[i])["F1"])
    return {"diff": macro(g, p1)["F1"] - macro(g, p2)["F1"], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_le_0": float(np.mean(np.array(d) <= 0))}


def ece(g, p, pr, bins=10):
    conf = pr.max(1); correct = (g == p).astype(float); e = 0.0; rel = []
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b) if a > 0 else (conf >= a) & (conf <= b)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean()); rel.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(e), rel


def risk_coverage(g, p, pr):
    order = np.argsort(-pr.max(1)); correct = (g == p)[order].astype(float)
    cov = np.arange(1, len(g) + 1) / len(g); acc = np.cumsum(correct) / np.arange(1, len(g) + 1)
    return cov, acc


out = {"models": {}}
tags = {"mixed": "Fine-tuned (real + synthetic)", "real_only": "Fine-tuned (real only)", "synth_only": "Fine-tuned (synthetic only)",
        "nli_deberta": "DeBERTa-v3-small NLI (zero-shot)", "base": "Qwen3-1.7B base (zero-shot)", "seed43": "Fine-tuned (seed 43)", "fewshot": "Qwen3-1.7B base (6-shot)", "deberta_ft": "DeBERTa-v3-small fine-tuned"}
P = {t: load(t) for t in tags}
for t, name in tags.items():
    if P[t] is None: continue
    out["models"][t] = {"name": name, "sets": {}}
    for s in SETS:
        g, p, pr = arr(P[t][s])
        e, rel = ece(g, p, pr)
        out["models"][t]["sets"][s] = {"n": len(g), "ci": boot(g, p), "ece": e, "reliability": rel,
                                         "per_class": macro(g, p)}
# selective prediction + per-kind for the main model
if P["mixed"]:
    out["selective"] = {}
    for s in SETS:
        g, p, pr = arr(P["mixed"][s]); cov, acc = risk_coverage(g, p, pr)
        pts = {}
        for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            pts[str(c)] = float(acc[int(round(c * len(g))) - 1])
        out["selective"][s] = pts
    out["per_kind"] = {}
    for s in SETS[:2]:
        d = {}
        for r in P["mixed"][s]:
            d.setdefault(r["src"].replace("medquad:", ""), []).append(r["gold"] == r["pred"])
        out["per_kind"][s] = {k: [float(np.mean(v)), len(v)] for k, v in d.items()}
# paired comparisons on HealthVer and OOD vs NLI baseline / other ablations
out["paired"] = {}
for other in ("nli_deberta", "real_only", "synth_only", "base", "fewshot", "deberta_ft", "seed43"):
    if P["mixed"] and P[other]:
        for s in SETS:
            g, p1, _ = arr(P["mixed"][s]); _, p2, _ = arr(P[other][s])
            out["paired"][f"mixed_vs_{other}/{s}"] = paired_diff(g, p1, p2)
(R / "analysis_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for t, m in out["models"].items():
    for s, v in m["sets"].items():
        c = v["ci"]
        print(f"{t:12s}{s:18s} acc {c['acc']['point']:.3f} F1 {c['F1']['point']:.3f} [{c['F1']['lo']:.3f},{c['F1']['hi']:.3f}] "
              f"P {c['P']['point']:.3f} R {c['R']['point']:.3f} ECE {v['ece']:.3f}")
print(json.dumps(out.get("selective"), indent=0)); print(json.dumps(out.get("per_kind"), indent=0))
for k, v in out["paired"].items(): print(k, {a: round(b, 3) for a, b in v.items()})
