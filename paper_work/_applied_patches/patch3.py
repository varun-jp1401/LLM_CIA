p = r"c:\LLM\LLM_CIA\paper_work\build_paper.py"
b = open(p, encoding="utf8").read()


def rep(old, new):
    global b
    assert old in b, "missing: " + old[:80]
    b = b.replace(old, new, 1)


# ---- generic drop for seed table
rep('''if not HAVE_ABL:
    _drop("abl")''', '''HAVE_SEED = "seed43" in M
if not HAVE_ABL:
    _drop("abl")
if not HAVE_SEED:
    _drop("seed")''')

# ---- intro citation
rep("are trained on general-domain text, and little is reported", "are trained on general-domain text [@laban2022,honovich2022], and little is reported")

# ---- related work subsection
rep('P.h2("Self-Correction and Selective Repair")', '''P.h2("NLI-Based Consistency Checking and Selective Prediction under Shift")
P.body("Natural language inference is the usual core of automatic factual-consistency checks. FactCC trained a classifier on synthetic data obtained by applying rule-based transformations to source sentences [@kryscinski2020], "
       "which is the idea behind our synthetic claim generator; SummaC showed that NLI models applied to whole documents underperform unless the input is segmented at sentence granularity [@laban2022], a choice mirrored by the sentence-level claim and evidence handling in our pipeline; "
       "TRUE re-evaluated factual-consistency metrics across a collection of datasets [@honovich2022]; and AlignScore trained a unified alignment function on a large multi-task collection [@zha2023]. "
       "These works address summarisation and general text rather than medical claims. Faithfulness to a source and factuality with respect to world knowledge are distinct properties [@maynez2020]; the verifier studied here judges the former, relative to a retrieved passage. "
       "For medical language models, Med-HALT offers a test set for hallucination [@pal2023]. A recurring concern with synthetic training data is that rule-generated errors may not resemble real ones, which is the reason we evaluate on topics, edit types and real claims withheld from training.")
P.body("Abstention is only useful if confidence scores are trustworthy under distribution shift. Kamath *et al.* proposed training a calibrator to decide when a question-answering model should abstain under domain shift [@kamath2020], "
       "and Desai and Durrett studied the calibration of pre-trained Transformers in and out of domain [@desai2020]. "
       "General-purpose results on calibration [@guo2017] and selective classification [@geifman2017] have rarely been checked for a medical claim verifier trained on mixed synthetic and real data, which the analyses in Section V-E and V-F provide.")
P.h2("Self-Correction and Selective Repair")''')

# ---- seed variability
rep('''else:
    P.body("[ablation pending]")''', '''    if HAVE_SEED:
        sc = lambda tag, s: M[tag]["sets"][s]["ci"]["F1"]["point"]
        rows = []
        for tag, nm in (("mixed", "42 (reported)"), ("seed43", "43")):
            rows.append([nm] + [f"{sc(tag, s):.3f}" for s in SETS])
        rows.append(["Mean"] + [f"{(sc('mixed', s) + sc('seed43', s)) / 2:.3f}" for s in SETS])
        rows.append(["Absolute difference"] + [f"{abs(sc('mixed', s) - sc('seed43', s)):.3f}" for s in SETS])
        P.table("seed", "Training-Seed Variability of the Fine-Tuned Verifier (Macro-F1)",
                ["Seed", "MQ-iid", "MQ-ood", "HealthVer"], rows, [3.0, 1.8, 1.8, 1.8], font=6.6, bold_rows=(2,),
                footnote="Seed 43 changes the sampled training rows, adapter initialisation and example order; hyper-parameters are unchanged.")
        dmax = max(abs(sc("mixed", s) - sc("seed43", s)) for s in SETS)
        P.body(f"A second training run with a different seed (which also redraws the 4,000 HealthVer and 2,500 synthetic training rows) changes macro-F1 by at most {dmax:.3f} on the three test sets ({{T:seed}}). "
               "With only two runs this is a rough indication, not a variance estimate, but it is smaller than the differences between the training-data configurations and between the verifier and the baselines, and it is of the same order as the bootstrap intervals.")
else:
    P.body("[ablation pending]")''')

# ---- extended-results appendix
rep('P.h5("Data and Code Availability")', '''P.h5("Appendix: Extended Results")
TAGS = [("mixed", "Fine-tuned"), ("real_only", "Real only"), ("synth_only", "Synth. only"), ("nli_deberta", "DeBERTa NLI"), ("base", "Qwen base")]
TAGS = [(t, n) for t, n in TAGS if t in M]
rows = []
for tag, nm in TAGS:
    for s, sn in zip(SETS, ["MQ-iid", "MQ-ood", "HealthVer"]):
        c = M[tag]["sets"][s]["ci"]
        f = lambda k: f"{c[k]['point']:.3f} [{c[k]['lo']:.2f},{c[k]['hi']:.2f}]"
        rows.append([nm if s == SETS[0] else "", sn, f("acc"), f("P"), f("R")])
P.table("full", "Accuracy and Macro Precision/Recall with 95% Bootstrap Intervals for All Systems",
        ["System", "Set", "Accuracy", "Macro-P", "Macro-R"], rows, [1.5, 1.0, 2.0, 2.0, 2.0], font=5.6, align=["l", "l", "c", "c", "c"],
        shade_rows=tuple(range(0, 3)))
LABL = ["entailment", "neutral", "contradiction"]
PRD = {t: json.loads((RES / f"predictions_{t}.json").read_text(encoding="utf-8")) for t, _ in TAGS}


def class_f1(rows_, k):
    tp = sum(r["pred"] == k and r["gold"] == k for r in rows_)
    fp = sum(r["pred"] == k and r["gold"] != k for r in rows_)
    fn = sum(r["pred"] != k and r["gold"] == k for r in rows_)
    pr_, rc_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return 2 * pr_ * rc_ / max(1e-9, pr_ + rc_)


rows = []
for tag, nm in TAGS:
    for s, sn in zip(SETS, ["MQ-iid", "MQ-ood", "HealthVer"]):
        rows.append([nm if s == SETS[0] else "", sn] + [f"{class_f1(PRD[tag][s], k):.3f}" for k in LABL])
P.table("pcf1", "Per-Class F1 for All Systems", ["System", "Set", "Entailment", "Neutral", "Contradiction"], rows, [1.9, 1.3, 1.7, 1.7, 1.9], font=6.0,
        align=["l", "l", "c", "c", "c"], shade_rows=tuple(range(0, 3)))
P.body("The extended tables confirm the pattern of the main text. The unadapted backbone and the NLI baseline fail mostly on the neutral and entailment classes of real claims, whereas the adapters fail mostly on contradiction under unseen edit types; "
       "the real-only adapter is the weakest on the synthetic sets and the synthetic-only adapter is the weakest on real claims.")

P.h5("Data and Code Availability")''')
open(p, "w", encoding="utf8").write(b)
print("patched3")
