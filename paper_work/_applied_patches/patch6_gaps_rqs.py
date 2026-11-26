"""Align the paper's research gaps and research questions with the five gaps (G1-G5) stated in the project PPT."""
import shutil

p = r"C:\LLM\LLM_CIA\paper_work\build_paper.py"
shutil.copy(p, p + ".bak")
s = open(p, encoding="utf8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)


# --- abstract
rep('''    "Precision and recall of 90% or more are therefore reached''',
    '''    "Of the project\\u2019s five research gaps, claim-level verification and uncertainty handling are evaluated and safety-oriented evaluation is addressed at verifier level; risk-aware assessment and selective repair are implemented in the prototype but not evaluated. "
    "Precision and recall of 90% or more are therefore reached''')

# --- introduction: the five gaps and the research questions
a = s.index('P.body("We therefore ask three questions.')
b = s.index('P.body("The main contributions are as follows.")')
new = '''P.body("The project proposal condenses these limitations into five research gaps. **G1, claim-level verification:** individual claims, not only whole responses, must be checked against evidence. "
       "**G2, risk-aware assessment:** errors must be distinguished by their potential clinical severity. **G3, selective repair:** high-risk errors must be corrected while accurate content is preserved. "
       "**G4, uncertainty handling:** claims that cannot be reliably verified must be recognised so that unsupported corrections are avoided. **G5, safety-oriented evaluation:** correction must be evaluated together with clinical risk reduction and unintended errors. "
       "This paper addresses G1 and G4 empirically and G5 at the level of the verifier, that is, the detection of unsupported claims and the acceptance of contradicted ones. "
       "G2 and G3 are implemented in the prototype pipeline of Section III-B (a keyword risk screen; grounded repair with re-verification) but are not evaluated here, and no claim is made about their effectiveness.")
P.body("We therefore ask three questions, each tied to a gap. **RQ1 (G1):** can a 1.7-billion-parameter open model, adapted with LoRA on a single consumer GPU, verify individual medical claims against evidence better than a zero-shot NLI cross-encoder, "
       "and how does it fare on held-out topics, unseen error types and real claims? **RQ2 (G5):** how reliably does it detect unsupported and contradicted claims, how often does it accept a contradicted claim, and which training data (synthetic, real or both) determine that behaviour? "
       "**RQ3 (G4):** how reliable are the model\\u2019s confidence scores, and what accuracy can be obtained by abstaining on low-confidence cases? The work was motivated by a target of at least 90% precision and recall; we report where that target is and is not met.")
'''
s = s[:a] + new + s[b:]

# --- contributions tagged with the gaps they serve
rep('and withholds one contradiction family and several overclaim templates for out-of-distribution testing.")',
    'and withholds one contradiction family and several overclaim templates for out-of-distribution testing (G1).")')
rep('against a zero-shot DeBERTa-v3 NLI verifier and the unadapted backbone.")',
    'against a zero-shot DeBERTa-v3 NLI verifier and the unadapted backbone (G1).")')
rep('and real-claim performance.")',
    'and real-claim performance, with unsupported-claim detection and accepted-contradiction rates (G1, G5).")')
rep('together with an explicit account of why the 90% target is met on synthetic but not on real claims.")',
    'together with an explicit account of why the 90% target is met on synthetic but not on real claims (G4).")')

# --- related work: observations + gap-coverage table
a = s.index('P.body("Three observations follow from {T:rel}.')
b = s.index('# ======================================================================= III. METHODOLOGY')
new_rw = '''P.body("Three observations follow from {T:rel}. First, most prior studies evaluate or filter the outputs of large (frequently proprietary) models rather than train a compact, locally deployable verifier. "
       "Second, in the abstracts examined, performance is not reported separately for topic-held-out, unseen-edit and real-claim conditions, so the synthetic-to-real gap of a trained verifier is not visible. "
       "Third, calibration is reported for verbalised confidence of large models [@boie2026] but, in these works, not for the token-level confidence of a small adapted verifier, even though that confidence "
       "determines when a system should abstain. Together with the five gaps G1\\u2013G5 stated in the Introduction, these observations define the research gap addressed here; {T:gaps} states which of the gaps this paper treats and how far.")
P.table("gaps", "Research Gaps of the Project Proposal and Their Treatment in This Paper",
        ["Gap", "Treatment in this paper", "Status"],
        [["G1 Claim-level verification", "LoRA-adapted Qwen3-1.7B verifier vs. NLI and prompting; held-out topics, unseen edits, real claims", "Evaluated (RQ1)"],
         ["G2 Risk-aware assessment", "Keyword risk screen in the pipeline (Algorithm 1)", "Implemented; not evaluated"],
         ["G3 Selective repair", "Grounded repair with re-verification in the pipeline", "Implemented; four-question walk-through only"],
         ["G4 Uncertainty handling", "Calibration, temperature scaling and coverage\\u2013accuracy of the verifier", "Verifier evaluated (RQ3); pipeline abstention rule not evaluated"],
         ["G5 Safety-oriented evaluation", "Flag recall, false-flag and accepted-contradiction rates at \\u03c4 = 0.70", "Verifier level only (RQ2); repair success, unnecessary repair and risk reduction not measured"]],
        [1.3, 3.0, 2.2], font=6.2, align=["l", "l", "l"])

'''
s = s[:a] + new_rw + s[b:]

# --- detection section answers RQ2
rep('"{T:det} evaluates that binary decision', '"{T:det} addresses RQ2 by evaluating that binary decision')

# --- answers to the research questions
a = s.index('P.bullet(f"**RQ1.** Yes, with qualifications.')
b = s.index('P.h2("Discussion and Threats to Validity")')
new_ans = '''P.bullet(f"**RQ1 (G1, claim-level verification).** Yes, with qualifications. The adapted 1.7B verifier exceeds the zero-shot NLI verifier on all three test sets (macro-F1 {ci('mixed','medquad_test_iid')}, {ci('mixed','medquad_test_ood')}, {ci('mixed','healthver_test')} against {ci('nli_deberta','medquad_test_iid')}, {ci('nli_deberta','medquad_test_ood')}, {ci('nli_deberta','healthver_test')}), "
         + ("but a small encoder fine-tuned on the same data is a competitive alternative (see the comparison above). " if "deberta_ft" in M else "")
         + "The evidence supports the value of in-domain supervised adaptation; it does not show that a decoder LLM is required for verification.")
P.bullet(f"**RQ2 (G5, safety-oriented evaluation at verifier level).** At \\u03c4 = 0.70 on real claims the adapted verifier flags {DT['mixed']['healthver_test']['flag_recall']:.1%} of unsupported claims, wrongly flags {DT['mixed']['healthver_test']['false_flag_rate']:.1%} of entailed ones and lets {DT['mixed']['healthver_test']['contra_accepted']:.1%} of contradicted claims pass as supported "
         f"(zero-shot NLI: {DT['nli_deberta']['healthver_test']['flag_recall']:.1%}, {DT['nli_deberta']['healthver_test']['false_flag_rate']:.1%} and {DT['nli_deberta']['healthver_test']['contra_accepted']:.1%}). "
         f"The synthetic-to-real gap is large ({ci('mixed','medquad_test_iid')} \\u2192 {ci('mixed','medquad_test_ood')} \\u2192 {ci('mixed','healthver_test')} from held-out topics to unseen edits to real claims). "
         + (f"Synthetic-only training collapses on real claims ({ci('synth_only','healthver_test')}) and real-only training is weak on MedQuAD ({ci('real_only','medquad_test_iid')}, {ci('real_only','medquad_test_ood')}), so both sources are needed. " if HAVE_ABL else "")
         + "Repair success, unnecessary-repair rate and clinical risk reduction, the remaining parts of G5, were not measured.")
P.bullet(f"**RQ3 (G4, uncertainty handling).** Confidence is over-confident on real claims (ECE {M['mixed']['sets']['healthver_test']['ece']:.3f}) but can be repaired with a little real validation data" + (f" (ECE {CAL['healthver_test']['ece_after']:.3f} after temperature scaling)" if HAVE_CAL else "")
         + f". Abstaining on the least confident half of real claims lifts accuracy to {S['healthver_test']['0.5']:.3f}; it does not reach 0.90.")
P.bullet("**Gaps not answered (G2, G3).** Risk-aware assessment and selective repair are present in the prototype pipeline (a keyword risk screen and a grounded repair step with re-verification) but were not evaluated, "
         "so this study cannot say whether they identify clinically significant errors, repair them successfully, or preserve accurate content. They are the first priority for future work.")

'''
s = s[:a] + new_ans + s[b:]

# --- conclusion
rep('+ "The study thus meets the 90% precision and recall target only in controlled synthetic conditions.")',
    '+ "Of the five research gaps set out in the proposal, the study evaluates G1 and G4 and, at the level of the verifier, G5; G2 (risk-aware assessment) and G3 (selective repair) are implemented in the prototype but not evaluated. "\n'
    '       "The study thus meets the 90% precision and recall target only in controlled synthetic conditions.")')
open(p, "w", encoding="utf8").write(s)

q = r"C:\LLM\LLM_CIA\paper_work\build_lib.py"
l = open(q, encoding="utf8").read()
assert 'ORDER_T = ["rel", "edits"' in l
l = l.replace('ORDER_T = ["rel", "edits"', 'ORDER_T = ["rel", "gaps", "edits"')
open(q, "w", encoding="utf8").write(l)
print("patch6 applied")
