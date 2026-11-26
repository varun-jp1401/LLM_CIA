"""Paper figures at IEEE single-column width (3.45 in). Reads results/*.json written by the experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
A = json.loads((RES / "analysis_summary.json").read_text(encoding="utf-8"))
LOG = json.loads((ROOT / "models" / "qwen3-1.7b-lora-claimverify" / "train_log.json").read_text(encoding="utf-8"))["log"]

BLUE, ORANGE, GREEN, VERM, GREY, SKY, PINK, INK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8A8A8A", "#56B4E9", "#CC79A7", "#222222"
LAB = ["entailment", "neutral", "contradiction"]
SETS = ["medquad_test_iid", "medquad_test_ood", "healthver_test"]
SETN = ["MedQuAD\nheld-out topics", "MedQuAD\n+ unseen edits", "HealthVer\n(real)"]
W = 3.45
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.frameon": False, "legend.fontsize": 7,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=INK, lw=0.8, fs=6.3, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, p, q, lw=0.8, color=INK, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ Fig: system architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(W, 4.9))
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1); ax.axis("off")
    IMPL, EVAL = "#F2F2F2", "#DCEBF7"
    box(ax, .10, .925, .50, .06, "User medical question", fc="#FFFFFF")
    arrow(ax, (.35, .925), (.35, .895))
    box(ax, .05, .83, .60, .065, "Draft generation: Qwen3-1.7B\n(greedy decoding)", fc=IMPL)
    arrow(ax, (.35, .83), (.35, .80))
    box(ax, .05, .735, .60, .065, "Claim extraction\n(sentence-level decomposition)", fc=IMPL)
    arrow(ax, (.35, .735), (.35, .705))
    box(ax, .05, .60, .60, .105, "Hybrid evidence retrieval\nBM25-style lexical + PubMedBERT dense (FAISS)\nfused by reciprocal rank fusion\nover MedQuAD / PubMedQA", fc=IMPL, fs=5.6)
    arrow(ax, (.35, .60), (.35, .565))
    # verification container
    ax.add_patch(FancyBboxPatch((.02, .385), .96, .18, boxstyle="round,pad=0.004,rounding_size=0.012", fc="#FFFFFF", ec=INK, lw=0.9))
    ax.text(.5, .545, "Claim verification:  (claim, evidence)  →  entailed / neutral / contradicted", fontsize=6.1, ha="center", va="center")
    box(ax, .045, .40, .425, .115, "Deployed verifier\nDeBERTa-v3-small NLI\n(zero-shot cross-encoder)", fc=IMPL, fs=5.9)
    box(ax, .53, .40, .425, .115, "This work\nQwen3-1.7B + LoRA claim\nverifier (stand-alone)", fc=EVAL, ec=BLUE, lw=1.4, fs=5.9)
    ax.text(.5, .457, "or", fontsize=6, ha="center", va="center", color=GREY)
    arrow(ax, (.5, .385), (.5, .35))
    box(ax, .12, .285, .76, .065, "Risk screening (keyword rules: dose, drug, emergency, ...)", fc=IMPL, fs=5.8)
    arrow(ax, (.22, .285), (.17, .245)); arrow(ax, (.5, .285), (.5, .245)); arrow(ax, (.78, .285), (.83, .245))
    box(ax, .02, .155, .30, .09, "Supported\n→ accept draft", fc="#FFFFFF", fs=5.8)
    box(ax, .35, .155, .30, .09, "Unsupported,\nnot high-risk\n→ grounded repair", fc="#FFFFFF", fs=5.6)
    box(ax, .68, .155, .30, .09, "Unsupported,\nhigh-risk\n→ abstain", fc="#FFFFFF", fs=5.6)
    arrow(ax, (.5, .155), (.5, .125))
    box(ax, .30, .075, .40, .05, "Re-verification of repaired text", fc=IMPL, fs=5.7)
    arrow(ax, (.5, .075), (.5, .05))
    arrow(ax, (.17, .155), (.30, .052), lw=0.6); arrow(ax, (.83, .155), (.70, .052), lw=0.6)
    box(ax, .20, 0.0, .60, .05, "Final answer / flagged answer / abstention", fc="#FFFFFF", fs=5.8)
    ax.add_patch(FancyBboxPatch((.69, .945), .04, .03, boxstyle="round,pad=0.002", fc=EVAL, ec=BLUE, lw=1.2))
    ax.text(.745, .96, "evaluated here", fontsize=5.5, va="center")
    ax.add_patch(FancyBboxPatch((.69, .895), .04, .03, boxstyle="round,pad=0.002", fc=IMPL, ec=INK, lw=0.6))
    ax.text(.745, .91, "implemented,\nnot evaluated", fontsize=5.5, va="center", linespacing=1.0)
    save(fig, "fig_architecture")


# ------------------------------------------------------------------ Fig: data + training pipeline
def fig_dataflow():
    fig, ax = plt.subplots(figsize=(W, 3.7))
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1); ax.axis("off")
    G = "#F2F2F2"; E = "#DCEBF7"
    box(ax, .02, .875, .45, .11, "MedQuAD (NIH QA corpus)\n5,477 documents, 9 sources", fc=G)
    box(ax, .53, .875, .45, .11, "HealthVer (real claims)\n5,005 / 850 / 840 rows", fc=G)
    arrow(ax, (.245, .875), (.245, .84)); arrow(ax, (.755, .875), (.755, .84))
    box(ax, .02, .715, .45, .125, "Topic-disjoint split by Focus\n80 / 10 / 10 %\n4,372 / 561 / 544 documents", fc=G, fs=6)
    box(ax, .53, .715, .45, .125, "Label mapping\nSupports → entailment\nRefutes → contradiction\nNeutral → neutral", fc=G, fs=5.8)
    arrow(ax, (.245, .715), (.245, .68))
    box(ax, .02, .395, .45, .285, "Synthetic claim generator\n• evidence window, 2–4 sentences\n• entailment: meaning-\n   preserving edits\n• contradiction: negation,\n   antonym, quantifier, number\n• neutral: same / other\n   document, overclaim", fc=G, fs=5.6)
    arrow(ax, (.245, .395), (.245, .355)); arrow(ax, (.755, .715), (.755, .355))
    box(ax, .02, .20, .96, .155, "", fc="#FFFFFF")
    ax.text(.5, .325, "Training mixture", fontsize=6.4, fontweight="bold", ha="center")
    ax.text(.5, .275, "2,500 synthetic + 4,000 HealthVer train = 6,500 examples", fontsize=6.0, ha="center")
    ax.text(.5, .225, "validation: 300 synthetic + 450 HealthVer dev (model selection only)", fontsize=5.7, ha="center", color=GREY)
    arrow(ax, (.5, .20), (.5, .165))
    box(ax, .07, 0.015, .86, .15, "LoRA fine-tuning of Qwen3-1.7B (r = 16, all linear layers)\nfirst-answer-token loss; best checkpoint by validation macro-F1", fc=E, ec=BLUE, lw=1.2, fs=5.8)
    save(fig, "fig_dataflow")

# ------------------------------------------------------------------ Fig: validation curve
def fig_curve():
    import re as _re
    loss = []
    for l in (RES / "train_claimverify_main.log").read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _re.search(r"\[train\] step (\d+)/\d+ loss ([0-9.]+)", l)
        if m:
            loss.append((int(m.group(1)), float(m.group(2))))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15))
    ax = axes[0]
    ax.plot([a for a, _ in loss], [b for _, b in loss], color=VERM, lw=1.0)
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Training loss"); ax.set_yscale("log"); ax.grid(lw=0.3, alpha=0.4)
    ax.set_title("Training loss", fontsize=8)
    ax = axes[1]
    steps = [e["step"] for e in LOG]
    for key, lab, c, m in (("macro_f1", "F1", BLUE, "o"), ("macro_precision", "P", ORANGE, "s"), ("macro_recall", "R", GREEN, "^")):
        ax.plot(steps, [e[key] for e in LOG], marker=m, ms=2.8, lw=1.0, color=c, label=lab)
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Validation (macro)"); ax.set_ylim(0.68, 0.88)
    ax.grid(axis="y", lw=0.3, alpha=0.5); ax.legend(loc="lower right", ncol=3, columnspacing=0.8, handlelength=1.2)
    ax.set_title("Validation", fontsize=8)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_curve")


# ------------------------------------------------------------------ Fig: main comparison with CIs
def grouped_ci(tags, names, colors, hatches, fname, h=2.3, ymin=0.0, legend_cols=2):
    fig, ax = plt.subplots(figsize=(W, h))
    n = len(tags); w = 0.8 / n
    for j, t in enumerate(tags):
        if t not in A["models"]:
            continue
        pts, lo, hi = [], [], []
        for s in SETS:
            c = A["models"][t]["sets"][s]["ci"]["F1"]
            pts.append(c["point"]); lo.append(c["point"] - c["lo"]); hi.append(c["hi"] - c["point"])
        x = np.arange(3) + (j - (n - 1) / 2) * w
        ax.bar(x, pts, w * 0.92, yerr=[lo, hi], color=colors[j], hatch=hatches[j], edgecolor="white", lw=0.4,
               error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK), label=names[j])
        for xi, v in zip(x, pts):
            ax.text(xi, min(v + 0.045, 1.02), f"{v:.2f}", ha="center", fontsize=5.3, rotation=90 if n > 3 else 0)
    ax.axhline(0.9, color=GREY, lw=0.7, ls="--", zorder=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(SETN, fontsize=6.3)
    ax.set_ylim(ymin, 1.16); ax.set_ylabel("Macro-F1 (95% bootstrap CI)"); ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.legend(loc="upper center", ncol=legend_cols, bbox_to_anchor=(0.5, 1.22))
    save(fig, fname)


# ------------------------------------------------------------------ Fig: confusion matrices
def fig_confusion():
    P = json.loads((RES / "predictions_mixed.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.45))
    for ax, s, nm in zip(axes, SETS, ["MedQuAD", "MedQuAD+unseen", "HealthVer"]):
        M = np.zeros((3, 3))
        for r in P[s]:
            M[LAB.index(r["gold"]), LAB.index(r["pred"])] += 1
        Mn = M / M.sum(1, keepdims=True)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{Mn[a, b]:.2f}", ha="center", va="center", fontsize=5.8, color="white" if Mn[a, b] > .55 else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(["E", "N", "C"], fontsize=6.3)
        ax.set_yticks(range(3)); ax.set_yticklabels(["E", "N", "C"], fontsize=6.3)
        ax.set_title(nm, fontsize=6.5, pad=2); ax.set_xlabel("Predicted", fontsize=6.3, labelpad=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("True", fontsize=6.3, labelpad=1)
    fig.tight_layout(pad=0.4, w_pad=0.4)
    save(fig, "fig_confusion")


# ------------------------------------------------------------------ Fig: calibration + selective prediction
def fig_calibration():
    P = json.loads((RES / "predictions_mixed.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.95))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--")
    cols = [BLUE, ORANGE, VERM]
    for s, c, nm in zip(SETS, cols, ["MedQuAD", "+unseen", "HealthVer"]):
        rel = A["models"]["mixed"]["sets"][s]["reliability"]
        xs = [r[0] for r in rel]; ys = [r[1] for r in rel]
        ax.plot(xs, ys, marker="o", ms=2.5, lw=0.9, color=c, label=f"{nm} (ECE {A['models']['mixed']['sets'][s]['ece']:.3f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_xlim(0.3, 1.02); ax.set_ylim(0.3, 1.02)
    ax.legend(loc="upper left", fontsize=5.4); ax.set_title("Reliability", fontsize=7.5)
    ax = axes[1]
    for s, c, nm in zip(SETS, cols, ["MedQuAD", "+unseen", "HealthVer"]):
        rows = P[s]
        conf = np.array([max(r["probs"]) for r in rows]); ok = np.array([r["gold"] == r["pred"] for r in rows], float)
        o = np.argsort(-conf); acc = np.cumsum(ok[o]) / np.arange(1, len(o) + 1); cov = np.arange(1, len(o) + 1) / len(o)
        ax.plot(cov, acc, lw=1.0, color=c, label=nm)
    ax.axhline(0.9, color=GREY, lw=0.7, ls="--")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Accuracy on retained"); ax.set_ylim(0.55, 1.02); ax.set_xlim(0.05, 1.0)
    ax.set_title("Selective prediction", fontsize=7.5)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_calibration")


# ------------------------------------------------------------------ Fig: per-edit-type accuracy
def fig_kinds():
    K = A["per_kind"]
    names = {"entail": "Entailment", "negation": "Contra.: negation", "antonym": "Contra.: antonym", "quantifier": "Contra.: quantifier",
             "number": "Contra.: number", "swap_ood": "Contra.: demog./direct. (unseen)",
             "neutral_same_doc": "Neutral: same doc.", "neutral_other_doc": "Neutral: other doc.", "neutral_overclaim": "Neutral: overclaim"}
    order = ["entail", "negation", "antonym", "quantifier", "number", "swap_ood", "neutral_same_doc", "neutral_other_doc", "neutral_overclaim"]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ys = np.arange(len(order))[::-1]
    for j, (s, c, h, lab) in enumerate((("medquad_test_iid", BLUE, "", "held-out topics"), ("medquad_test_ood", ORANGE, "//", "+ unseen edit types"))):
        v = [K[s].get(k, [np.nan, 0])[0] for k in order]
        ax.barh(ys + (0.2 if j == 0 else -0.2), v, 0.36, color=c, hatch=h, edgecolor="white", lw=0.4, label=lab)
        for y, val, k in zip(ys, v, order):
            if not np.isnan(val):
                ax.text(val + 0.01, y + (0.2 if j == 0 else -0.2), f"{val:.2f}", va="center", fontsize=5.3)
    ax.set_yticks(ys); ax.set_yticklabels([names[k] for k in order], fontsize=6.2)
    ax.axvline(0.9, color=GREY, lw=0.7, ls="--"); ax.set_xlim(0, 1.12); ax.set_xlabel("Accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=2, fontsize=7)
    save(fig, "fig_kinds")


# ------------------------------------------------------------------ Fig: PR curves for unsupported-claim detection
def fig_pr():
    from sklearn.metrics import precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.0), sharey=True)
    specs = [("mixed", "Fine-tuned (ours)", BLUE, "-"), ("nli_deberta", "DeBERTa-v3 NLI", ORANGE, "--"), ("base", "Qwen3-1.7B base", GREY, ":")]
    for ax, s, nm in zip(axes, ["medquad_test_ood", "healthver_test"], ["MedQuAD + unseen edits", "HealthVer (real)"]):
        for tag, lab, c, ls in specs:
            p = RES / f"predictions_{tag}.json"
            if not p.exists():
                continue
            rows = json.loads(p.read_text(encoding="utf-8"))[s]
            y = np.array([r["gold"] != "entailment" for r in rows]); sc = 1 - np.array([r["probs"][0] for r in rows])
            pr, rc, _ = precision_recall_curve(y, sc)
            ax.plot(rc, pr, color=c, ls=ls, lw=1.1, label=lab)
            flagged = sc >= 0.30
            tp = (flagged & y).sum()
            ax.plot([tp / max(1, y.sum())], [tp / max(1, flagged.sum())], marker="o", ms=3.5, color=c, mec=INK, mew=0.4, zorder=5)
        ax.axhline(0.9, color=GREY, lw=0.5, ls="--"); ax.axvline(0.9, color=GREY, lw=0.5, ls="--")
        ax.set_xlabel("Recall (unsupported)"); ax.set_title(nm, fontsize=7); ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.02)
        ax.grid(lw=0.3, alpha=0.4)
    axes[0].set_ylabel("Precision"); axes[0].legend(loc="lower left", fontsize=5.4)
    fig.tight_layout(pad=0.5)
    save(fig, "fig_pr")


if __name__ == "__main__":
    fig_pr()
    fig_architecture(); fig_dataflow(); fig_curve(); fig_confusion(); fig_calibration(); fig_kinds()
    tags_ = [t for t in ["mixed", "deberta_ft", "nli_deberta", "fewshot", "base"] if t in A["models"]]
    nm_ = {"mixed": "Qwen3-1.7B + LoRA (ours)", "deberta_ft": "DeBERTa-v3 fine-tuned", "nli_deberta": "DeBERTa-v3 NLI zero-shot", "fewshot": "Qwen3-1.7B 6-shot", "base": "Qwen3-1.7B zero-shot"}
    cl_ = {"mixed": BLUE, "deberta_ft": GREEN, "nli_deberta": ORANGE, "fewshot": SKY, "base": GREY}
    ht_ = {"mixed": "", "deberta_ft": "xx", "nli_deberta": "//", "fewshot": "\\\\", "base": ".."}
    grouped_ci(tags_, [nm_[t] for t in tags_], [cl_[t] for t in tags_], [ht_[t] for t in tags_], "fig_main", legend_cols=2, h=2.7)
    grouped_ci(["real_only", "synth_only", "mixed"], ["Real only", "Synthetic only", "Mixed (ours)"],
               [VERM, SKY, BLUE], ["//", "..", ""], "fig_ablation", legend_cols=3, h=2.3)
