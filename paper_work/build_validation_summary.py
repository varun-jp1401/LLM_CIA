"""Research Validation Summary (Deliverable 4): what is claimed, what supports it, what is missing."""
import json
from pathlib import Path
import docx
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
A = json.loads((RES / "analysis_summary.json").read_text(encoding="utf-8"))
M = A["models"]
SETS = ["medquad_test_iid", "medquad_test_ood", "healthver_test"]


def f1(tag, s):
    c = M[tag]["sets"][s]["ci"]["F1"]
    return f"{c['point']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


d = docx.Document()
sec = d.sections[0]
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = d.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(10.5)


def H(t, lvl=1):
    h = d.add_heading(t, level=lvl)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)


def P(t, bold=False, italic=False):
    p = d.add_paragraph()
    r = p.add_run(t); r.bold = bold; r.italic = italic
    p.paragraph_format.space_after = Pt(4)
    return p


def B(t):
    p = d.add_paragraph(style="List Bullet")
    p.add_run(t)
    p.paragraph_format.space_after = Pt(2)


def T(header, rows, widths):
    t = d.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
        for r in t.rows[0].cells[i].paragraphs[0].runs:
            r.bold = True; r.font.size = Pt(9.5)
    for ri, row in enumerate(rows, 1):
        for ci, v in enumerate(row):
            t.rows[ri].cells[ci].text = str(v)
            for r in t.rows[ri].cells[ci].paragraphs[0].runs:
                r.font.size = Pt(9.5)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
    d.add_paragraph()


title = d.add_heading("Research Validation Summary", 0)
P("Accompanies: Evidence-Grounded Claim Verification for Medical Language-Model Outputs (IEEE-format manuscript). "
  "This document states what the paper claims, what supports each claim, and what is missing or assumed. It is intended for the authors and supervisor, not for publication.", italic=True)

H("1. Research gaps (from the project PPT) and how the paper treats them")
P("The proposal slides (slides 6-7) list five research gaps. The paper now states them as G1-G5 in the Introduction and ties each research question to one of them (Table II of the paper).")
T(["Gap (PPT slide 7)", "Treatment in the paper", "Status / evidence"],
  [["G1 Claim-level verification", "LoRA-adapted Qwen3-1.7B verifier vs. zero-shot NLI, prompting and a fine-tuned encoder; held-out topics, unseen edits, real claims (RQ1)", "Evaluated, with bootstrap CIs"],
   ["G2 Risk-aware assessment", "Keyword regex risk screen inside pipeline.py", "Implemented, NOT evaluated or clinically validated"],
   ["G3 Selective repair", "Grounded repair + re-verification in pipeline.py; no repair-specific training", "Implemented, NOT evaluated (four-question walk-through only)"],
   ["G4 Uncertainty handling", "ECE, temperature scaling, coverage-accuracy of the verifier (RQ3)", "Verifier evaluated; pipeline abstention rule (keyword-based) not evaluated"],
   ["G5 Safety-oriented evaluation", "Flag recall, false-flag and accepted-contradiction rates at tau=0.70 (RQ2)", "Verifier level only; repair success, unnecessary repair and risk reduction not measured"]],
  [4.0, 7.2, 5.4])
B("Research questions: RQ1 (G1) verification vs baselines across held-out topics / unseen edits / real claims; RQ2 (G5) detection of unsupported claims, accepted contradictions and the role of training data; RQ3 (G4) calibration and abstention.")
B("The paper makes no claim about G2 and G3. Closing them needs an end-to-end evaluation (high-risk error recall, repair success rate, unnecessary-repair rate, post-repair unsupported rate, safe-abstention rate) with reference annotations.")
B("Literature-based gap (supporting G1/G4): in the abstracts examined, performance is not reported separately for topic-held-out, unseen-edit and real-claim conditions, and calibration is reported for verbalised confidence of large models but not for the token-level confidence of a small adapted verifier. This is based on abstracts and the authors' reading; it is not a systematic review.")

H("2. Principal contributions and their evidence")
T(["Contribution", "Evidence in the paper", "Strength"],
  [["Topic-disjoint synthetic claim generator with a withheld contradiction family", "build_claim_dataset.py; leakage assertion; Tables II, III; MQ-ood set", "Implemented and checked (no topic overlap asserted in code)"],
   ["LoRA-adapted Qwen3-1.7B claim verifier vs zero-shot NLI and base model", f"MQ-iid {f1('mixed','medquad_test_iid')}, MQ-ood {f1('mixed','medquad_test_ood')}, HealthVer {f1('mixed','healthver_test')}; NLI baseline {f1('nli_deberta','healthver_test')} on HealthVer", "Supported with bootstrap CIs; single training seed unless the seed table is present"],
   ["Ablation of training-data composition", "Real-only vs synthetic-only vs mixed (Table VII)", "Supported; unequal training sizes acknowledged; checkpoint selected on a validation set that includes real rows"],
   ["Calibration and selective prediction analysis", "ECE, reliability, coverage-accuracy, validation-based temperature scaling and thresholds", "Supported for HealthVer and MedQuAD test sets; no clinical risk weighting"],
   ["Documented negative PubMedQA result", "Table of the 152-item test split; majority-class baseline 0.546", "Reported from earlier project runs (results/eval_pubmedqa_*.json); not re-run in this study"]],
  [5.0, 7.0, 4.6])

H("3. Novelty claims and what they do and do not say")
B("The paper does NOT claim to be the first claim-verification study in medicine, nor that a 1.7B decoder is better than encoders in general. It claims a specific, documented evaluation of a LoRA-adapted small LLM under topic-held-out and real-claim conditions.")
B("The use of rule-based synthetic edits to train a verifier follows prior work (e.g. FactCC, cited); the contribution is the topic-disjoint, family-withheld evaluation and the combination with real claims, not the idea of synthetic edits.")

H("4. Main experimental findings")
T(["System", "MQ-iid macro-F1", "MQ-ood macro-F1", "HealthVer macro-F1"],
  [[M[t]["name"]] + [f1(t, s) for s in SETS] for t in M],
  [5.2, 3.8, 3.8, 3.8])
P("Targets: the original goal was at least 90% precision and recall. It is met on the synthetic sets only. On real HealthVer claims the best system reaches macro-precision/recall in the mid-0.70s; "
  "this must not be presented as meeting the goal.", bold=True)

H("5. Contradictions and inconsistencies found in the project materials")
B("Planned vs implemented adaptation: the proposal slides specify QLoRA (4-bit NF4), r=8, q_proj and v_proj only (about 1.61 M trainable parameters). The implementation uses plain LoRA in bf16, r=16, on all seven projection types (17.43 M parameters). The paper states the implemented configuration.")
B("GPU: README.md reports an RTX 5050 Laptop GPU for earlier runs; nvidia-smi on the machine used for the claim-verification experiments reports an RTX 4050 Laptop GPU (6 GB). The paper reports the RTX 4050 for the claim-verification runs.")
B("PubMedQA base-model result: two result files exist for the same 152-item test split. results/eval_pubmedqa_base_test.json shows an all-'yes' collapse (accuracy 0.546 = the majority-class rate), while results/eval_pubmedqa_balanced_test.json reports 0.461 for the base model. "
  "The paper reports the latter in the table and the majority-class baseline separately. The cause of the difference was not investigated.")
B("README statistics: it reports 11,274 MedQuAD XML files and a '15,000-document' FAISS index; the claim-verification corpus uses 5,477 documents that contain usable answers (nine of the twelve collections retain answer text). "
  "The README's 71.16% 'token accuracy' is next-token accuracy, not task accuracy, and is not used in the paper.")
B("Pipeline vs adapter: pipeline.py verifies claims with the DeBERTa NLI model; the fine-tuned Qwen verifier is not wired into pipeline.verify(). The paper therefore evaluates the Qwen verifier stand-alone and says so.")
B("HealthVer domain: 99.8% of rows concern COVID-19, so 'real claims' here also means 'a different medical domain from MedQuAD'.")

H("6. Missing experiments and limitations")
B("End-to-end evaluation of the pipeline (repair success, unnecessary-repair rate, post-repair unsupported rate, safe-abstention rate, clinical risk reduction) was not performed; the keyword risk screen is unvalidated.")
B("No clinical-expert review or human evaluation of generated or repaired text.")
B("Single training seed for the main model (a second seed is reported only if the seed table appears in the paper); confidence intervals reflect test-set sampling, not training variance.")
B("Only one real claim dataset (HealthVer re-split); no MedNLI, SciFact or similar; no claims generated by the LLM itself.")
B("Retrieval quality (BM25/dense/RRF) was not evaluated; the verifier is tested on given evidence, not on retrieved evidence.")
B("The fine-tuned encoder baseline is a small cross-encoder; larger biomedical encoders (e.g. PubMedBERT-size or larger) were not tried.")
B("QLoRA, larger backbones and longer training schedules were not tested; validation macro-F1 was still rising at the last step.")

H("7. Assumptions made")
B("Author list taken from the project slides (three students, one supervisor); affiliation, university, city and e-mail fields are placeholders.")
B("HealthVer labels Supports/Refutes/Neutral were mapped to entailment/contradiction/neutral.")
B("The re-split HealthVer mirror on the Hugging Face hub was used as published; its construction was not independently audited beyond verifying zero claim overlap between train and test.")
B("The licence of the HealthVer mirror and the redistribution terms of MedQuAD (CC BY 4.0 per its readme) should be checked before releasing data.")

H("8. Claims that need further validation before submission")
B("Any statement that the verifier is 'safe' or 'clinically useful' is unsupported; the paper does not make it.")
B("The causal explanation of why the PubMedQA formulation failed (small training set, mixed objective, task mismatch) is plausible but was not tested by controlled experiments.")
B("The explanation of the MQ-ood contradiction errors (unfamiliar edit types) is consistent with the per-type results but was not isolated by an additional ablation.")

H("9. Reference verification notes")
B("Every reference was checked against Crossref (journals, ACL-anthology papers) or the arXiv API (arXiv preprints); titles, authors, venues, years, pages and DOIs in the reference list are generated from those records.")
B("Reddy and Samreen (Sci. Rep., 2026): the DOI and bibliographic record exist in Crossref, but no abstract or full text could be retrieved, so the paper cites it only for what its title states. Please read the paper before relying on it.")
B("Robertson and Zaragoza (BM25 monograph): Crossref's volume/page data disagreed with the commonly cited print record, so volume and pages were omitted and only the DOI is given.")
B("BioBERT is dated 2020 (volume 36, issue 4) although it appeared online in 2019; De Busser et al. is dated 2025 (print volume 47) although Crossref lists the 2024 online-first date.")
B("Several 2026 references (Boie, Liang, Reddy) are recent; verify that they are still accessible and unchanged at submission time.")
B("Hugging Face model and dataset cards are cited as online resources (HTTP 200 verified).")

d.save(ROOT / "paper" / "Research_Validation_Summary.docx")
print("saved validation summary")
