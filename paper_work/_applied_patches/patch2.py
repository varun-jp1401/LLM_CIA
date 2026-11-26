p = r"c:\LLM\LLM_CIA\paper_work\build_paper.py"
b = open(p, encoding="utf8").read()


def rep(old, new):
    global b
    assert old in b, "missing: " + old[:80]
    b = b.replace(old, new, 1)


# --- remove test-mode hack, add generic dropping of numbering for missing items
i = b.index("import build_lib\nif not HAVE_ABL:")
j = b.index("P = Paper(")
b = b[:i] + '''import build_lib
HAVE_CAL = (RES / "analysis_calib.json").exists()


def _drop(name):
    for d in (build_lib.TABN, build_lib.FIGN):
        if name in d:
            v = d.pop(name)
            for k in d:
                if d[k] > v:
                    d[k] -= 1


if not HAVE_ABL:
    _drop("abl")
if not HAVE_CAL:
    _drop("op")
''' + b[j:]

# --- simplified supported() equation
i = b.index('P.equation(R("supported", True)')
j = b.index("\n", i)
b = b[:i] + 'P.equation(R("supported", True) + Paren(R("c")) + R(" = ") + R("1") + Paren(R("max", True) + Sub(R(""), R("e")) + R(" ") + Sub(R("p"), R("NLI", True)) + Paren(R("entail", True) + R(" | e, c")) + R(" ≥ 0.70"), "[", "]"))' + b[j:]

# --- curve caption
rep('"Validation macro-precision, macro-recall and macro-F1 during training (450 HealthVer-dev and 300 synthetic validation rows)."',
    '"Training loss (left, log scale) and validation macro-F1, precision and recall (right; 450 HealthVer-dev and 300 synthetic rows) during training."')

# --- recalibration subsection + appendix
rep('P.h2("Breakdown by Source Collection and Length")', '''if HAVE_CAL:
    CAL = json.loads((RES / "analysis_calib.json").read_text(encoding="utf-8"))["mixed"]
    P.h2("Validation-Based Recalibration and Operating Points")
    P.body("The preceding analyses used the fixed threshold of the deployed pipeline. This subsection fits recalibration parameters and thresholds on validation data only (HealthVer development rows for HealthVer, MedQuAD validation rows for the synthetic sets) and applies them unchanged to the test sets. "
           "Temperature scaling [@guo2017] divides the log class probabilities by a single scalar *T* fitted by minimising validation negative log-likelihood; it does not change predictions or accuracy. "
           "Two operating points are then chosen for the unsupported-claim decision: the threshold that maximises validation F1 for flagging unsupported claims, and the smallest threshold that accepts at most 5% of contradicted validation claims (a safety-oriented rule).")
    rows = []
    for s, sn in zip(SETS, ["MQ-iid", "MQ-ood", "HealthVer"]):
        d = CAL[s]
        for k, lab in (("fixed_0.70", "fixed 0.70"), ("f1_opt", "max val. F1"), ("safety_5pct", "\u22645% C accepted")):
            x = d[k]
            rows.append([sn if k == "fixed_0.70" else "", lab, f"{x['tau']:.2f}", f"{x['flag_recall']:.3f}", f"{x['flag_precision']:.3f}", f"{x['false_flag_rate']:.3f}", f"{x['contra_accepted']:.3f}"])
    P.table("op", "Operating Points Selected on Validation Data (Fine-Tuned Verifier, Test Results)",
            ["Test set", "Rule", "\u03c4", "Flag R", "Flag P", "False-flag", "C accepted"],
            rows, [1.5, 2.0, 0.7, 1.0, 1.0, 1.2, 1.2], font=6.4, align=["l", "l", "c", "c", "c", "c", "c"],
            footnote=f"Temperature scaling: T = {CAL['medquad_test_iid']['T']:.2f} (MedQuAD validation) and {CAL['healthver_test']['T']:.2f} (HealthVer dev). ECE before \u2192 after scaling: MQ-iid {CAL['medquad_test_iid']['ece_before']:.3f} \u2192 {CAL['medquad_test_iid']['ece_after']:.3f}, "
                     f"MQ-ood {CAL['medquad_test_ood']['ece_before']:.3f} \u2192 {CAL['medquad_test_ood']['ece_after']:.3f}, HealthVer {CAL['healthver_test']['ece_before']:.3f} \u2192 {CAL['healthver_test']['ece_after']:.3f}.")
    h = CAL["healthver_test"]
    P.body(f"Temperature scaling reduces the HealthVer calibration error from {h['ece_before']:.3f} to {h['ece_after']:.3f} (fitted T = {h['T']:.2f}), which confirms that the over-confidence in {{F:cal}} is largely a matter of scale and can be corrected with a small amount of real validation data. "
           f"Choosing the threshold on validation data gives, on real test claims, flag recall {h['f1_opt']['flag_recall']:.3f} and precision {h['f1_opt']['flag_precision']:.3f} for the F1-optimal rule (false-flag rate {h['f1_opt']['false_flag_rate']:.3f}), "
           f"and {h['safety_5pct']['contra_accepted']:.3f} accepted contradictions with a false-flag rate of {h['safety_5pct']['false_flag_rate']:.3f} for the safety rule. "
           "The numbers show the trade-off an operator faces: protecting against accepted contradictions costs unnecessary repairs, and neither rule reaches 90% on both flag precision and flag recall for real claims.")

P.h2("Breakdown by Source Collection and Length")''')

rep('P.h5("Data and Code Availability")', '''P.h5("Appendix: Prompt, Generation Settings and Risk Rules")
P.body("For reproducibility we reproduce the verification prompt and the pipeline settings used by the deployed system.")
P.algorithm("Verification prompt (user turn; the answer begins after the generation prompt)", [
    Raw("Classify whether the evidence supports, is neutral to, or contradicts the claim. Reply with one word: entailment, neutral, or contradiction."),
    Raw("Claim: <claim>"), Raw("Evidence: <evidence, first 1,800 characters>"),
    Raw("Target token: first token of entailment | neutral | contradiction (thinking disabled)")])
P.algorithm("Pipeline settings and keyword risk rules (pipeline.py)", [
    Raw("Draft: greedy decoding, max 240 new tokens; repair: max 180 new tokens"),
    Raw("Retrieval: k = 5 for verification, top 3 passages for repair"),
    Raw("NLI input: 12 evidence sentences of highest lexical overlap, max length 256"),
    Raw("High-risk terms: dose, dosage, prescription, drug interaction, overdose, stop taking, start taking, emergency, diagnosis of, ACE inhibitor(s), beta blocker(s), diuretic(s), acetaminophen, ibuprofen, naproxen, antibiotic(s)")])

P.h5("Data and Code Availability")''')

rep('statistics (analysis.py, analysis_extra.py)', 'statistics (analysis.py, analysis_extra.py, analysis_calib.py)')
open(p, "w", encoding="utf8").write(b)
print("patched2")
