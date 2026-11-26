"""Generate publication figures (PDF vector + 300-dpi PNG) from the evaluation artefacts.

    python make_figures.py          ->  figures/fig1..fig5 (.pdf/.png) + results_table.tex/.csv
Inputs: results/eval_claim_verification.json, models/qwen3-1.7b-lora-claimverify/train_log.json,
        data/claim_verification/manifest.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
R = json.loads((ROOT / "results" / "eval_claim_verification.json").read_text(encoding="utf-8"))
LOG = json.loads((ROOT / "models" / "qwen3-1.7b-lora-claimverify" / "train_log.json").read_text(encoding="utf-8"))["log"]
MAN = json.loads((ROOT / "data" / "claim_verification" / "manifest.json").read_text(encoding="utf-8"))

# Okabe-Ito colour-blind-safe palette; hatches keep it readable in greyscale print.
BLUE, ORANGE, GREEN, VERM, GREY, INK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8A8A8A", "#222222"
LABELS = ("entailment", "neutral", "contradiction")
SETS = [("medquad_test_iid", "MedQuAD\nheld-out topics"),
        ("medquad_test_ood", "MedQuAD\nheld-out topics +\nunseen edit types"),
        ("healthver_test", "HealthVer\n(real claims)")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 150,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def thresh(ax, y=0.9):
    ax.axhline(y, color=GREY, lw=0.8, ls="--", zorder=0)


# ------------------------------------------------ Fig 1: validation curve
steps = [e["step"] for e in LOG]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for key, lab, c, m in (("macro_f1", "Macro-F1", BLUE, "o"), ("macro_precision", "Macro precision", ORANGE, "s"),
                       ("macro_recall", "Macro recall", GREEN, "^")):
    ax.plot(steps, [e[key] for e in LOG], marker=m, ms=4, lw=1.4, color=c, label=lab)
best = max(LOG, key=lambda e: e["macro_f1"])
ax.annotate(f"best {best['macro_f1']:.3f}", (best["step"], best["macro_f1"]), xytext=(-48, -22),
            textcoords="offset points", fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax.set_xlabel("Optimizer step"); ax.set_ylabel("Validation score")
ax.set_ylim(0.65, 0.9); ax.grid(axis="y", lw=0.3, alpha=0.5)
ax.legend(loc="lower right")
ax.set_title("Validation during LoRA fine-tuning", loc="left")
save(fig, "fig1_validation_curve")

# ------------------------------------------------ Fig 2: adapter vs base, P / R / F1
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
metrics = (("macro_precision", "Precision"), ("macro_recall", "Recall"), ("macro_f1", "F1"))
cols, hatches = (BLUE, ORANGE, GREEN), ("", "//", "..")
x = np.arange(2)
for ax, (key, title) in zip(axes, metrics):
    pass
for i, (sk, sname) in enumerate(SETS):
    ax = axes[i]
    w = 0.26
    for j, (mk, mname) in enumerate(metrics):
        vals = [R["adapter"][sk][mk]] + ([R["base_zero_shot"][sk][mk]] if sk in R["base_zero_shot"] else [np.nan])
        bars = ax.bar(np.arange(2) + (j - 1) * w, vals, w * 0.92, color=cols[j], hatch=hatches[j],
                      edgecolor="white", lw=0.5, label=mname if i == 0 else None)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(["Fine-tuned", "Base\n(zero-shot)"])
    ax.set_title(sname, fontsize=8)
    ax.set_ylim(0, 1.08); ax.grid(axis="y", lw=0.3, alpha=0.5)
    thresh(ax)
axes[0].set_ylabel("Macro-averaged score")
from matplotlib.lines import Line2D
h, l = axes[0].get_legend_handles_labels()
h.append(Line2D([0], [0], color=GREY, lw=0.8, ls="--")); l.append("90% target")
fig.legend(h, l, loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.02))
fig.suptitle("Claim verification: fine-tuned vs. base Qwen3-1.7B", x=0.01, y=1.02, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig2_adapter_vs_base")

# ------------------------------------------------ Fig 3: normalised confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
short = ["Entail.", "Neutral", "Contra."]
for ax, (sk, sname) in zip(axes, SETS):
    cm = R["adapter"][sk]["confusion"]
    M = np.array([[cm[g][p] for p in LABELS] for g in LABELS], float)
    Mn = M / M.sum(1, keepdims=True)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{Mn[a, b]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if Mn[a, b] > 0.55 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_title(sname.replace("\n", " "), fontsize=7.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
axes[0].set_ylabel("True")
fig.suptitle("Row-normalised confusion matrices (fine-tuned model)", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
save(fig, "fig3_confusion_matrices")

# ------------------------------------------------ Fig 4: per-class precision / recall
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
for ax, (sk, sname) in zip(axes, SETS):
    pc = R["adapter"][sk]["per_class"]
    xx = np.arange(3); w = 0.36
    for j, (k, c, h, lab) in enumerate((("precision", BLUE, "", "Precision"), ("recall", ORANGE, "//", "Recall"))):
        v = [pc[l][k] for l in LABELS]
        bars = ax.bar(xx + (j - 0.5) * w, v, w * 0.92, color=c, hatch=h, edgecolor="white", lw=0.5,
                      label=lab if sk == SETS[0][0] else None)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{val:.2f}", ha="center", fontsize=6)
    ax.set_xticks(xx); ax.set_xticklabels(short, fontsize=7)
    ax.set_title(sname.replace("\n", " "), fontsize=7.5)
    ax.set_ylim(0, 1.08); ax.grid(axis="y", lw=0.3, alpha=0.5)
    thresh(ax)
axes[0].set_ylabel("Score")
h, l = axes[0].get_legend_handles_labels()
h.append(Line2D([0], [0], color=GREY, lw=0.8, ls="--")); l.append("90% target")
fig.legend(h, l, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.02))
fig.suptitle("Per-class precision and recall (fine-tuned model)", x=0.01, y=1.02, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig4_per_class")

# ------------------------------------------------ Fig 5: generalisation gap (F1 vs. realism)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
order = ["medquad_test_iid", "medquad_test_ood", "healthver_test"]
names = ["Held-out\ntopics", "+ unseen\nedit types", "Real claims\n(HealthVer)"]
f1 = [R["adapter"][k]["macro_f1"] for k in order]
b = [R["base_zero_shot"][k]["macro_f1"] for k in order]
ax.plot(range(3), f1, marker="o", color=BLUE, lw=1.6, label="Fine-tuned")
ax.plot(range(3), b, marker="s", color=GREY, lw=1.2, ls="--", label="Base (zero-shot)")
for i, v in enumerate(f1):
    ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=7, color=BLUE)
for i, v in enumerate(b):
    ax.text(i, v - 0.07, f"{v:.3f}", ha="center", fontsize=7, color=GREY)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7)
ax.set_ylim(0.2, 1.08); ax.set_xlim(-0.3, 2.3); ax.set_ylabel("Macro-F1")
ax.grid(axis="y", lw=0.3, alpha=0.5); thresh(ax)
ax.text(2.28, 0.915, "90% target", ha="right", fontsize=7, color=GREY)
ax.legend(loc="lower left")
ax.set_title("Performance vs. test-set realism", loc="left")
save(fig, "fig5_generalisation")

# ------------------------------------------------ Fig 6: synthetic dataset composition
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
for j, (nm, c) in enumerate((("train", "#4D4D4D"), ("val", "#8A8A8A"), ("test_iid", "#B8B8B8"), ("test_ood", "#B8B8B8"))):
    ax.bar(j, MAN[nm]["n"], color=c, width=0.65, edgecolor="white")
    ax.text(j, MAN[nm]["n"] + 80, str(MAN[nm]["n"]), ha="center", fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels(["train", "val", "test\n(iid)", "test\n(ood)"], fontsize=7)
ax.set_ylabel("Examples (1/3 per class)"); ax.set_title("Synthetic split sizes", loc="left", fontsize=9)
ax.set_ylim(0, 6800)
ax = axes[1]
kinds = ["entail", "negation", "antonym", "quantifier", "number", "swap_ood",
         "neutral_same_doc", "neutral_other_doc", "neutral_overclaim"]
disp = ["Entailment", "Negation", "Antonym", "Quantifier", "Number", "Demographic/\ndirectional (OOD)",
        "Neutral: same doc", "Neutral: other doc", "Neutral: overclaim"]
tot = {k: sum(MAN[s]["kinds"].get(k, 0) for s in MAN) for k in kinds}
colors = [GREEN] + [VERM] * 5 + [BLUE] * 3
ax.barh(range(len(kinds))[::-1], [tot[k] for k in kinds], color=colors, edgecolor="white", height=0.7)
ax.set_yticks(range(len(kinds))[::-1]); ax.set_yticklabels(disp, fontsize=7)
for i, k in zip(range(len(kinds))[::-1], kinds):
    ax.text(tot[k] + 40, i, str(tot[k]), va="center", fontsize=7)
ax.set_xlabel("Examples (all splits)"); ax.set_title("Construction type (green: entail, red: contradict, blue: neutral)",
                                                      loc="left", fontsize=8)
fig.tight_layout()
save(fig, "fig6_dataset_composition")

# ------------------------------------------------ results table (csv + LaTeX)
rows = []
for who, key in (("Fine-tuned", "adapter"), ("Base (zero-shot)", "base_zero_shot")):
    for sk, sname in SETS:
        if sk in R.get(key, {}):
            m = R[key][sk]
            rows.append((who, sname.replace("\n", " "), m["n"], m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"]))
(OUT / "results_table.csv").write_text(
    "model,test_set,n,accuracy,macro_precision,macro_recall,macro_f1\n" +
    "\n".join(",".join(f'"{c}"' if isinstance(c, str) else str(c) for c in r) for r in rows), encoding="utf-8")
tex = ["\\begin{tabular}{llrrrrr}", "\\toprule",
       "Model & Test set & $n$ & Acc. & Macro-P & Macro-R & Macro-F1 \\\\", "\\midrule"]
for r in rows:
    tex.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]:.3f} & {r[4]:.3f} & {r[5]:.3f} & {r[6]:.3f} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(OUT / "results_table.tex").write_text("\n".join(tex), encoding="utf-8")
print("wrote results_table.{csv,tex}")
