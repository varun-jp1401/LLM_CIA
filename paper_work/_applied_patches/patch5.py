p = r"c:\LLM\LLM_CIA\paper_work\build_paper.py"
b = open(p, encoding="utf8").read()


def rep(old, new):
    global b
    assert old in b, "missing: " + old[:80]
    b = b.replace(old, new, 1)


# demo table numbering
rep('''if not HAVE_SEED:
    _drop("seed")''', '''if not HAVE_SEED:
    _drop("seed")
HAVE_DEMO = (RES / "pipeline_demo.json").exists()
if not HAVE_DEMO:
    _drop("demo")''')

# abstract: encoder baseline sentence
rep('''    + (f"Training-data ablations show''', '''    + (f"A DeBERTa-v3-small encoder fine-tuned on the same data reaches {ci('deberta_ft','healthver_test')} on real claims. " if "deberta_ft" in M else "")
    + (f"Training-data ablations show''')

# pipeline walk-through + RQ summary
rep('P.h2("Discussion and Threats to Validity")', '''if HAVE_DEMO:
    DEMO = json.loads((RES / "pipeline_demo.json").read_text(encoding="utf-8"))
    P.h2("Illustrative Walk-Through of the Deployed Pipeline")
    P.body("To show how the stages of Algorithm 1 behave together, the pipeline was run on four questions with the BM25-style retriever, the unadapted Qwen3-1.7B generator and the DeBERTa NLI verifier. "
           "This is a demonstration, not an evaluation: four questions cannot support any rate, and the corpus used for retrieval contains the answers to the first three. "
           "{T:demo} summarises the outcome of each run.")
    rows = []
    for d_ in DEMO:
        cl_ = d_["claims"]; uns = [c for c in cl_ if c["status"] == "unverified"]
        if d_["abstained"]:
            outc = "abstained"
        elif not uns:
            outc = "draft accepted"
        elif "Evidence check" in d_["answer"]:
            outc = "draft + warning"
        else:
            outc = "repaired"
        rows.append([Raw(d_["question"]), str(len(cl_)), str(len(cl_) - len(uns)), str(sum(c["risk"] == "high" for c in uns)), outc])
    P.table("demo", "Pipeline Outcomes on Four Illustrative Questions",
            ["Question", "Claims", "Supported", "High-risk unsupp.", "Outcome"], rows, [3.6, 0.9, 1.2, 1.4, 1.4], font=6.0, align=["l", "c", "c", "c", "c"],
            footnote="Supported: entailment score \\u2265 0.70 (Algorithm 1). Outcomes follow the policy in Algorithm 1.")
    allc = [(c, d_["question"]) for d_ in DEMO for c in d_["claims"]]
    if allc:
        n_sup = sum(c["status"] == "potentially_supported" for c, _ in allc)
        low = min(allc, key=lambda x: x[0]["confidence"])[0]
        hi = max(allc, key=lambda x: x[0]["confidence"])[0]
        P.body(f"Across the four runs, {n_sup} of {len(allc)} generated claims passed the 0.70 support threshold. The highest-scoring claim (entailment score {hi['confidence']:.2f}) was \\u201c{hi['claim'][:140]}\\u201d, "
               f"and the lowest-scoring (score {low['confidence']:.2f}) was \\u201c{low['claim'][:140]}\\u201d. "
               "Inspecting individual claims shows the behaviour expected from the quantitative results: a zero-shot NLI verifier is conservative about claims that paraphrase the evidence, which triggers the repair or warning path even for accurate drafts, "
               "and the keyword risk screen, not the verifier, decides which unsupported claims lead to abstention.")

P.h2("Answers to the Research Questions")
P.bullet(f"**RQ1.** Yes, with qualifications. The adapted 1.7B verifier exceeds the zero-shot NLI verifier on all three test sets (macro-F1 {ci('mixed','medquad_test_iid')}, {ci('mixed','medquad_test_ood')}, {ci('mixed','healthver_test')} against {ci('nli_deberta','medquad_test_iid')}, {ci('nli_deberta','medquad_test_ood')}, {ci('nli_deberta','healthver_test')}), "
         + ("but a small encoder fine-tuned on the same data is a competitive alternative (see the comparison above). " if "deberta_ft" in M else "")
         + "The evidence supports the value of in-domain supervised adaptation; it does not show that a decoder LLM is required for verification.")
P.bullet(f"**RQ2.** The synthetic-to-real gap is large: {ci('mixed','medquad_test_iid')} \\u2192 {ci('mixed','medquad_test_ood')} \\u2192 {ci('mixed','healthver_test')} as the test set moves from held-out topics to unseen edits to real claims. "
         + (f"Synthetic-only training collapses on real claims ({ci('synth_only','healthver_test')}) and real-only training is weak on MedQuAD ({ci('real_only','medquad_test_iid')}, {ci('real_only','medquad_test_ood')}), so both sources are needed. " if HAVE_ABL else ""))
P.bullet(f"**RQ3.** Confidence is over-confident on real claims (ECE {M['mixed']['sets']['healthver_test']['ece']:.3f}) but can be repaired with a little real validation data" + (f" (ECE {CAL['healthver_test']['ece_after']:.3f} after temperature scaling)" if HAVE_CAL else "")
         + f". Abstaining on the least confident half of real claims lifts accuracy to {S['healthver_test']['0.5']:.3f}; it does not reach 0.90.")

P.h2("Discussion and Threats to Validity")''')

# reproduction commands in appendix
rep('P.h5("Data and Code Availability")', '''P.algorithm("Commands that reproduce the reported results", [
    Raw("python build_claim_dataset.py --n-train 6000      # synthetic corpus"),
    Raw("python train_claim_verifier.py --n-real 4000 --n-synth 2500 --epochs 2"),
    Raw("python train_claim_verifier.py --n-synth 0 --out models/ablation-real-only"),
    Raw("python train_claim_verifier.py --n-real 0 --n-synth 2500 --out models/ablation-synth-only"),
    Raw("python dump_predictions.py --adapter <adapter|none> --tag <name> [--split val]"),
    Raw("python nli_baseline.py [val];  python dump_fewshot.py;  python train_encoder_baseline.py"),
    Raw("python analysis.py;  python analysis_extra.py;  python analysis_calib.py"),
    Raw("python paper_work/make_paper_figures.py;  python paper_work/build_paper.py")])

P.h5("Data and Code Availability")''')
open(p, "w", encoding="utf8").write(b)

# lib: add 'demo' ordering
q = r"c:\LLM\LLM_CIA\paper_work\build_lib.py"
l = open(q, encoding="utf8").read()
l = l.replace('"pq", "err", "full", "pcf1"]', '"pq", "err", "demo", "full", "pcf1"]')
open(q, "w", encoding="utf8").write(l)
print("patched5")
