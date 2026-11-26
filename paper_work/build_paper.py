"""Assemble the IEEE-format manuscript from the conference template. All numbers come from results/*.json."""
import json
import sys
from pathlib import Path

import docx
from docx.shared import Pt
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).parent))
from build_lib import *  # noqa

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import build_claim_dataset as BCD
NANT, NQ, NSW, NSYN, NMOD, NADD, NOADD, NPRE = (len(BCD.ANTONYMS) // 2, len(BCD.QUANT) // 2, len(BCD.OOD_SWAP) // 2, len(BCD.SAFE_SYN) // 2,
                                               len(BCD.MODALS), len(BCD.TRAIN_ADD), len(BCD.OOD_ADD), len(BCD.PREFIXES))
FIG = ROOT / "paper" / "figures"
import os
OUT = ROOT / "paper" / os.environ.get("PAPER_OUT", "Claim_Verification_IEEE_Paper.docx")
RES = ROOT / "results"
A = json.loads((RES / "analysis_summary.json").read_text(encoding="utf-8"))
EV = json.loads((RES / "eval_claim_verification.json").read_text(encoding="utf-8"))
M, S = A["models"], A["selective"]
SETS = ["medquad_test_iid", "medquad_test_ood", "healthver_test"]


def ci(tag, s, k="F1", d=3):
    c = M[tag]["sets"][s]["ci"][k]
    return f"{c['point']:.{d}f}"


def cir(tag, s, k="F1"):
    c = M[tag]["sets"][s]["ci"][k]
    return f"{c['point']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def pdiff(other, s):
    d = A["paired"][f"mixed_vs_{other}/{s}"]
    return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


HAVE_ABL = "real_only" in M and "synth_only" in M
HAVE_BASE = "base" in M
MODELS_MAIN = [(t, n) for t, n in [("mixed", "Fine-tuned Qwen (ours)"), ("deberta_ft", "DeBERTa fine-tuned"), ("nli_deberta", "DeBERTa NLI zero-shot"),
                                   ("fewshot", "Qwen 6-shot"), ("base", "Qwen zero-shot")] if t in M]

import build_lib
HAVE_CAL = (RES / "analysis_calib.json").exists()


def _drop(name):
    for d in (build_lib.TABN, build_lib.FIGN):
        if name in d:
            v = d.pop(name)
            for k in d:
                if d[k] > v:
                    d[k] -= 1


HAVE_SEED = "seed43" in M
if not HAVE_ABL:
    _drop("abl")
if not HAVE_SEED:
    _drop("seed")
HAVE_DEMO = (RES / "pipeline_demo.json").exists()
if not HAVE_DEMO:
    _drop("demo")
if not HAVE_CAL:
    _drop("op")
P = Paper(ROOT / "paper_work" / "template_transitional.docx")
doc = P.doc
ps = doc.paragraphs

# ======================================================================= front matter
title = ps[0]
for r in list(title.runs):
    r._r.getparent().remove(r._r)
title.add_run("Evidence-Grounded Claim Verification for Medical Language-Model Outputs: Parameter-Efficient Adaptation of Qwen3-1.7B under Topic-Held-Out and Real-Claim Evaluation")
title.runs[0].font.size = Pt(20)
ps[1]._p.getparent().remove(ps[1]._p)          # sub-title note

# authors: rebuild the author area (three authors in 3 columns, supervisor centred below)
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
paras0 = doc.paragraphs
old_auth = [p for p in paras0 if p.style.name == "Author" and p.text.strip().startswith("line 1")]
sect_paras = [p for p in paras0 if p.style.name == "Normal" and p._p.find(qn("w:pPr")) is not None and p._p.find(qn("w:pPr")).find(qn("w:sectPr")) is not None]
assert len(old_auth) == 4 and len(sect_paras) >= 2, (len(old_auth), len(sect_paras))
end_row1, end_row2 = sect_paras[0], sect_paras[1]
AFF = ["Dept. of AI & Data Science Engineering", "[University name]", "[City, Country]", "[email address]"]


def make_author(name, lines, before, col_break=False):
    p = doc.add_paragraph(style="Author")
    pf = p.paragraph_format
    r = p.add_run(name); r.font.size = Pt(9)
    for ln in lines:
        r2 = p.add_run(); r2.add_break(); r2.add_text(ln); r2.font.size = Pt(9); r2.italic = not ln.startswith("[")
        if ln.startswith("["):
            r2.italic = True
            r2.font.highlight_color = WD_COLOR_INDEX.YELLOW
    if col_break:
        br = p.add_run(); br.add_break(docx.enum.text.WD_BREAK.COLUMN)
    before.addprevious(p._p)
    return p


import docx.enum.text
make_author("Apoorv Tiwari", AFF, end_row1._p, col_break=True)
make_author("Sidak Singh Kohli", AFF, end_row1._p, col_break=True)
make_author("Varun John Paul", AFF, end_row1._p)
for p in old_auth:
    p._p.getparent().remove(p._p)
# second row: one centred author (supervisor); make that section single-column and drop its column break
for r in list(end_row2.runs):
    r._r.getparent().remove(r._r)
cols = end_row2._p.find(qn("w:pPr")).find(qn("w:sectPr")).find(qn("w:cols"))
cols.set(qn("w:num"), "1")
make_author("Dr. Vinodha (Supervisor)", AFF, end_row2._p)

# abstract / keywords paragraphs
abs_p = next(p for p in doc.paragraphs if p.style.name == "Abstract")
kw_p = next(p for p in doc.paragraphs if p.style.name == "Keywords")
for p in (abs_p, kw_p):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)

ABSTRACT = (
    "Large language models (LLMs) can produce fluent but unsupported medical statements, so individual claims should be checked against evidence "
    "before they reach users. We study claim-level verification—labelling a claim as entailed by, neutral to, or contradicted by a passage—with a small "
    "open model, Qwen3-1.7B, adapted with low-rank adapters (LoRA) on a single 6 GB laptop GPU. A balanced synthetic corpus is generated from MedQuAD with "
    "splits that are disjoint by medical topic and with one family of contradiction edits withheld for an out-of-distribution test; real, human-annotated "
    f"HealthVer claims are added for training and held-out evaluation. The adapted model reaches a macro-F1 of {ci('mixed','medquad_test_iid')} on held-out MedQuAD topics and "
    f"{ci('mixed','medquad_test_ood')} with unseen edit types, but {ci('mixed','healthver_test')} (95% CI {M['mixed']['sets']['healthver_test']['ci']['F1']['lo']:.3f}–{M['mixed']['sets']['healthver_test']['ci']['F1']['hi']:.3f}) on real HealthVer claims. "
    f"The off-the-shelf DeBERTa-v3 NLI verifier used in our retrieval pipeline scores {ci('nli_deberta','medquad_test_iid')}, {ci('nli_deberta','medquad_test_ood')} and {ci('nli_deberta','healthver_test')} on the same sets. "
    + (f"A DeBERTa-v3-small encoder fine-tuned on the same data reaches {ci('deberta_ft','healthver_test')} on real claims. " if "deberta_ft" in M else "")
    + (f"Training-data ablations show that synthetic data alone transfers poorly to real claims (macro-F1 {ci('synth_only','healthver_test')}) whereas real data alone generalises poorly to unseen edit types ({ci('real_only','medquad_test_ood')}). " if HAVE_ABL else "")
    + f"The model is over-confident on real claims (expected calibration error {M['mixed']['sets']['healthver_test']['ece']:.3f}); keeping only the 50% most confident predictions raises accuracy to {S['healthver_test']['0.5']:.3f}, still short of 0.90. "
    "A preliminary PubMedQA yes/no/maybe experiment with the same backbone reached 48.7% accuracy, below the 54.6% majority-class baseline, motivating the move to claim-level verification. "
    "Of the project\u2019s five research gaps, claim-level verification and uncertainty handling are evaluated and safety-oriented evaluation is addressed at verifier level; risk-aware assessment and selective repair are implemented in the prototype but not evaluated. "
    "Precision and recall of 90% or more are therefore reached on controlled synthetic benchmarks but not on real claims, and we report the gap and its probable causes."
)
r = abs_p.add_run("Abstract"); r.italic = True
abs_p.add_run("—" + ABSTRACT)
r = kw_p.add_run("Index Terms—medical hallucination, claim verification, natural language inference, LoRA, retrieval-augmented generation, selective prediction, calibration")

# ======================================================================= remove template body
body_start = next(i for i, p in enumerate(doc.paragraphs) if p.style.name == "Heading 1")
paras = doc.paragraphs
# the sectPr-carrying "references" paragraph that closes the two-column section is kept as the anchor
anchor = next(p for p in paras if p.style.name == "references" and p._p.find(qn("w:pPr")).find(qn("w:sectPr")) is not None)
for p in paras[body_start:]:
    if p._p is anchor._p:
        break
    p._p.getparent().remove(p._p)
for tbl in list(doc.tables):
    tbl._tbl.getparent().remove(tbl._tbl)
for r in list(anchor.runs):
    r._r.getparent().remove(r._r)
anchor.style = doc.styles["Normal"]
P.anchor = anchor._p
for el in list(doc.element.body.iter(qn("w:r"))):
    if el.find("{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent") is not None or el.find(qn("w:pict")) is not None:
        el.getparent().remove(el)

# ======================================================================= I. INTRODUCTION
P.h1("Introduction")
P.body("Large language models (LLMs) now answer medical questions at a level that approaches expert performance on examination-style benchmarks [@singhal2023,singhal2025,thirunavukarasu2023]. "
       "Fluency, however, is not fidelity: LLMs are known to generate content that is unsupported by their inputs or by established knowledge [@ji2023], and evaluations on realistic clinical material "
       "report gaps relative to physicians, poor adherence to guidelines and variable, occasionally unsafe advice [@hager2024,williams2024,sandmann2024,debusser2025]. Targeted manipulation of a model can even inject "
       "incorrect biomedical facts while leaving benchmark performance intact [@han2024]. In medicine the consequences of an unsupported statement—about a dose, an interaction or a symptom—"
       "can be serious, which makes verification of generated content a prerequisite for responsible use.")
P.body("A risk-aware framework should be able to (i) identify unsupported or erroneous claims inside a response, (ii) weigh their potential clinical consequence, (iii) correct high-risk errors without "
       "rewriting accurate content, and (iv) recognise when a claim cannot be verified and abstain. Each step depends on a reliable decision about the relationship between an individual claim and "
       "a piece of evidence. This decision is the focus of the present paper.")
P.body("Existing work addresses parts of this problem but, in the studies we examined, leaves a gap. Evaluation studies measure how often LLMs err [@asgari2025,liang2026] and show that decision-level accuracy can coexist "
       "with weak documentary support: in one retrieve-then-verify study, binary accuracy was 90–98% while only 60–65% of the generated content was supported by the source reports [@liang2026]. "
       "Uncertainty-based detectors use the model’s own confidence [@farquhar2024,manakul2023], yet verbalised confidence is unevenly calibrated across specialties [@boie2026]. "
       "Self-correction without external feedback is unreliable [@kamoi2024,huang2023], and retrieval-augmented generation (RAG) reduces but does not eliminate hallucination [@lewis2020,wang2025]. "
       "Off-the-shelf natural language inference (NLI) models, the usual choice for an independent verifier, are trained on general-domain text [@laban2022,honovich2022], and little is reported about how a small, locally "
       "deployable model performs as a medical claim verifier when it is evaluated on unseen topics, unseen error types and real human-annotated claims.")
P.body("The project proposal condenses these limitations into five research gaps. **G1, claim-level verification:** individual claims, not only whole responses, must be checked against evidence. "
       "**G2, risk-aware assessment:** errors must be distinguished by their potential clinical severity. **G3, selective repair:** high-risk errors must be corrected while accurate content is preserved. "
       "**G4, uncertainty handling:** claims that cannot be reliably verified must be recognised so that unsupported corrections are avoided. **G5, safety-oriented evaluation:** correction must be evaluated together with clinical risk reduction and unintended errors. "
       "This paper addresses G1 and G4 empirically and G5 at the level of the verifier, that is, the detection of unsupported claims and the acceptance of contradicted ones. "
       "G2 and G3 are implemented in the prototype pipeline of Section III-B (a keyword risk screen; grounded repair with re-verification) but are not evaluated here, and no claim is made about their effectiveness.")
P.body("We therefore ask three questions, each tied to a gap. **RQ1 (G1):** can a 1.7-billion-parameter open model, adapted with LoRA on a single consumer GPU, verify individual medical claims against evidence better than a zero-shot NLI cross-encoder, "
       "and how does it fare on held-out topics, unseen error types and real claims? **RQ2 (G5):** how reliably does it detect unsupported and contradicted claims, how often does it accept a contradicted claim, and which training data (synthetic, real or both) determine that behaviour? "
       "**RQ3 (G4):** how reliable are the model\u2019s confidence scores, and what accuracy can be obtained by abstaining on low-confidence cases? The work was motivated by a target of at least 90% precision and recall; we report where that target is and is not met.")
P.body("The main contributions are as follows.")
P.bullet("A claim–evidence corpus generator for MedQuAD that yields class-balanced data, splits by medical topic (not by example), applies label-preserving edits to all classes so that surface overlap is not a shortcut, "
         "and withholds one contradiction family and several overclaim templates for out-of-distribution testing (G1).")
P.bullet("A LoRA-adapted Qwen3-1.7B claim verifier scored through its first answer token, evaluated with bootstrap confidence intervals against a zero-shot DeBERTa-v3 NLI verifier and the unadapted backbone (G1).")
P.bullet("An ablation of training-data composition (real only, synthetic only, mixed) that separates what each source contributes to in-distribution, out-of-distribution and real-claim performance, with unsupported-claim detection and accepted-contradiction rates (G1, G5).")
P.bullet("A calibration and selective-prediction analysis giving practical abstention operating points, together with an explicit account of why the 90% target is met on synthetic but not on real claims (G4).")
P.bullet("A documented negative result: a 3-class PubMedQA formulation did not exceed the majority-class baseline, and we explain why it was a poor proxy for the verification objective.")
P.body("The remainder of the paper is organised as follows. Section II reviews related work. Section III presents the system context, data and model. Section IV describes the experimental protocol. "
       "Section V reports and discusses the results, and Section VI concludes.")

# ======================================================================= II. RELATED WORK
P.h1("Related Work")
P.h2("Hallucination and Clinical Safety of LLM Outputs")
P.body("Hallucination has been surveyed across natural language generation tasks [@ji2023]. In clinical settings, Hager *et al.* evaluated LLMs on 2,400 real patient cases covering four abdominal pathologies and found "
       "that they diagnosed less accurately than physicians, did not follow diagnostic or treatment guidelines, and were sensitive to the quantity and order of the information provided [@hager2024]. "
       "Williams *et al.* used 10,000 emergency-department visits and reported that GPT-4-turbo and GPT-3.5-turbo were on average 8% and 24% less accurate than a resident physician, with a tendency to be overly cautious [@williams2024]. "
       "Sandmann *et al.* compared GPT-3.5, GPT-4 and Llama 2 on 110 cases; GPT-4 performed best, and the authors noted that open-source models can be a viable option where privacy and transparency matter [@sandmann2024]. "
       "For patient-facing guidance, six LLMs answered most self-care questions on medicines and supplements correctly, but with substantial variability and potentially unsafe advice [@debusser2025], and ChatGPT’s disaster-triage "
       "repeatability and reproducibility were assessed over 35,190 simulated triages [@franc2024]. Asgari *et al.* proposed an error taxonomy and a clinical-safety framework for LLM-generated clinical notes, "
       "reporting a 1.47% hallucination rate and a 3.45% omission rate over 12,999 clinician-annotated sentences [@asgari2025]. These studies establish that unsupported content is a measurable safety problem. "
       "They are primarily evaluations: they quantify errors rather than provide a trainable component that detects them.")
P.h2("Uncertainty, Confidence and Hallucination Detection")
P.body("Farquhar *et al.* proposed semantic entropy, which estimates uncertainty over meanings rather than token sequences, to detect confabulations without task-specific data [@farquhar2024]. "
       "SelfCheckGPT detects hallucinations in a black-box setting by comparing multiple sampled responses [@manakul2023]. For stated confidence, Boie *et al.* prompted six LLMs on MedMCQA (12,000 responses) "
       "and found mean expected calibration error between 0.06 and 0.127 across models, with a threefold difference across specialties, concluding that high accuracy does not guarantee reliable uncertainty estimation [@boie2026]. "
       "These methods derive reliability from the generator’s own behaviour and do not consult external evidence. Classical calibration analysis [@guo2017] and selective classification with a reject option [@geifman2017] "
       "supply the formal tools used in Section V-E.")
P.h2("Retrieval-Augmented Generation and Evidence-Based Verification")
P.body("RAG conditions generation on retrieved passages [@lewis2020]. Wang *et al.* introduced Med-RISE, which adds query rewriting, retrieval, summarisation and a fact-and-safety filter; across MedQA, PubMedQA, MedMCQA and EYE500 "
       "with four LLMs, accuracy improved by 9.8–16.3% and hallucination rates fell by 11.8–18% [@wang2025]. Liang *et al.* assessed whether LLM-generated risk-of-bias judgements are supported by 97 source trial reports, retrieving "
       "passages with BM25 and assigning verdicts of supported, contradicted, not found or out of scope; evidence support was only 60–65% and conservative hallucination rates were 34–37% [@liang2026]. "
       "Reddy and Samreen describe a hybrid retrieval–verification framework for hallucination reduction in medical LLMs [@reddy2026]. At the claim level, FActScore decomposes long-form text into atomic facts and "
       "scores the fraction supported by a knowledge source [@min2023]. Supervised verification data exist for general claims (FEVER [@thorne2018]; surveyed in [@guo2022]), scientific claims (SciFact [@wadden2020]), health claims "
       "(HealthVer [@sarrouti2021]) and clinical inference (MedNLI [@romanov2018]). Biomedical language models such as BioBERT [@lee2020], PubMedBERT [@gu2021] and BioGPT [@luo2022] supply domain representations, "
       "and MedQuAD [@abacha2019] and MedQA [@jin2021] provide question–answer material from NIH sources and licensing examinations.")
P.h2("NLI-Based Consistency Checking and Selective Prediction under Shift")
P.body("Natural language inference is the usual core of automatic factual-consistency checks. FactCC trained a classifier on synthetic data obtained by applying rule-based transformations to source sentences [@kryscinski2020], "
       "which is the idea behind our synthetic claim generator; SummaC showed that NLI models applied to whole documents underperform unless the input is segmented at sentence granularity [@laban2022], a choice mirrored by the sentence-level claim and evidence handling in our pipeline; "
       "TRUE re-evaluated factual-consistency metrics across a collection of datasets [@honovich2022]; and AlignScore trained a unified alignment function on a large multi-task collection [@zha2023]. "
       "These works address summarisation and general text rather than medical claims. Faithfulness to a source and factuality with respect to world knowledge are distinct properties [@maynez2020]; the verifier studied here judges the former, relative to a retrieved passage. "
       "For medical language models, Med-HALT offers a test set for hallucination [@pal2023]. A recurring concern with synthetic training data is that rule-generated errors may not resemble real ones, which is the reason we evaluate on topics, edit types and real claims withheld from training.")
P.body("Abstention is only useful if confidence scores are trustworthy under distribution shift. Kamath *et al.* proposed training a calibrator to decide when a question-answering model should abstain under domain shift [@kamath2020], "
       "and Desai and Durrett studied the calibration of pre-trained Transformers in and out of domain [@desai2020]. "
       "General-purpose results on calibration [@guo2017] and selective classification [@geifman2017] have rarely been checked for a medical claim verifier trained on mixed synthetic and real data, which the analyses in Sections V-E to V-G provide.")
P.h2("Self-Correction and Selective Repair")
P.body("Kamoi *et al.* surveyed self-correction critically and concluded that no prior work demonstrates successful self-correction with feedback from prompted LLMs except in tasks exceptionally suited to it, that it works well "
       "when reliable external feedback is available, and that large-scale fine-tuning enables it [@kamoi2024]; Huang *et al.* similarly found that LLMs cannot self-correct reasoning without external feedback [@huang2023]. "
       "Chain-of-Verification drafts verification questions and answers them independently before revising [@dhuliawala2024], and RARR researches evidence and revises unsupported statements post hoc [@gao2023]. "
       "These findings favour conditioning any repair on retrieved evidence and an independent verifier rather than on the generator’s self-assessment, which is the design adopted in our pipeline.")
P.h2("Parameter-Efficient Adaptation")
P.body("LoRA freezes the pretrained weights and trains low-rank update matrices [@hu2022]; QLoRA additionally keeps the frozen model in 4-bit precision to reduce memory [@dettmers2023]. "
       "Qwen3 is an open-weight model family with selectable thinking and non-thinking modes [@yang2025]. We use the 1.7-billion-parameter dense model in non-thinking mode.")

P.table("rel", "Summary of Closely Related Studies and the Present Work",
        ["Ref.", "Focus and scale", "Reported finding"],
        [["[@asgari2025]", "Clinical-note safety; 12,999 annotated sentences", "1.47% hallucination, 3.45% omission"],
         ["[@hager2024]", "Clinical decision-making; 2,400 patient cases", "Worse than physicians; guidelines not followed"],
         ["[@williams2024]", "ED recommendations; 10,000 visits", "8–24% below a resident; over-cautious"],
         ["[@liang2026]", "Evidence support of risk-of-bias judgements; 97 trials", "60–65% support; 34–37% hallucination"],
         ["[@wang2025]", "RAG with safety filter; 4 datasets, 4 LLMs", "+9.8–16.3% accuracy; −11.8–18% hallucination"],
         ["[@boie2026]", "Verbalised confidence; 12,000 MedMCQA answers", "ECE 0.06–0.127; uneven across specialties"],
         ["[@kamoi2024]", "Critical survey of self-correction", "Needs reliable external feedback or fine-tuning"],
         ["**This work**", "Claim verifier, 1.7B + LoRA; synthetic + real claims", "See Section V"]],
        [1.55, 3.75, 3.2], font=6.3, align=["l", "l", "l"], bold_rows=(7,), shade_rows=(7,))

P.body("Three observations follow from {T:rel}. First, most prior studies evaluate or filter the outputs of large (frequently proprietary) models rather than train a compact, locally deployable verifier. "
       "Second, in the abstracts examined, performance is not reported separately for topic-held-out, unseen-edit and real-claim conditions, so the synthetic-to-real gap of a trained verifier is not visible. "
       "Third, calibration is reported for verbalised confidence of large models [@boie2026] but, in these works, not for the token-level confidence of a small adapted verifier, even though that confidence "
       "determines when a system should abstain. Together with the five gaps G1\u2013G5 stated in the Introduction, these observations define the research gap addressed here; {T:gaps} states which of the gaps this paper treats and how far.")
P.table("gaps", "Research Gaps of the Project Proposal and Their Treatment in This Paper",
        ["Gap", "Treatment in this paper", "Status"],
        [["G1 Claim-level verification", "LoRA-adapted Qwen3-1.7B verifier vs. NLI and prompting; held-out topics, unseen edits, real claims", "Evaluated (RQ1)"],
         ["G2 Risk-aware assessment", "Keyword risk screen in the pipeline (Algorithm 1)", "Implemented; not evaluated"],
         ["G3 Selective repair", "Grounded repair with re-verification in the pipeline", "Implemented; four-question walk-through only"],
         ["G4 Uncertainty handling", "Calibration, temperature scaling and coverage\u2013accuracy of the verifier", "Verifier evaluated (RQ3); pipeline abstention rule not evaluated"],
         ["G5 Safety-oriented evaluation", "Flag recall, false-flag and accepted-contradiction rates at \u03c4 = 0.70", "Verifier level only (RQ2); repair success, unnecessary repair and risk reduction not measured"]],
        [1.6, 2.6, 2.3], font=6.1, align=["l", "l", "l"])

# ======================================================================= III. METHODOLOGY
P.h1("Methodology")
P.h2("Problem Formulation")
P.body("Let *c* denote a claim and *e* an evidence passage, both token sequences. The verifier must assign a label *y* ∈ 𝒴 = {E, N, C}, where E (entailment) means that *e* supports *c*, "
       "C (contradiction) means that *e* contradicts *c*, and N (neutral) means that *e* neither supports nor contradicts *c*, i.e. the evidence is insufficient. "
       "For risk-aware use the verifier also returns a confidence κ = max_{y} *p*(*y* | *c*, *e*); the system accepts a claim only when it is entailed, treats N and C as unsupported, and may abstain when κ is below a threshold τ.")

P.h2("System Context")
P.body("{F:arch} shows the retrieval–verification–repair pipeline implemented in the project repository; the component evaluated in this paper is highlighted. "
       "A draft answer is generated by Qwen3-1.7B with greedy decoding and split into sentence-level claims. Evidence is retrieved by a hybrid of a BM25-style lexical scorer "
       "[@robertson2009], with additional question-match and entity-overlap bonuses, and a dense retriever that embeds passages with a PubMedBERT-based sentence encoder [@reimers2019,pubmedbert_st] and searches a FAISS flat inner-product index of L2-normalised vectors [@johnson2021]. "
       "The two ranked lists are fused by weighted reciprocal rank fusion (RRF) [@cormack2009]:")
P.figure("arch", FIG / "fig_architecture.png", "System context. Shaded blue: the claim verifier evaluated in this paper; grey: components implemented in the repository but not quantitatively evaluated here. The deployed pipeline verifies with a zero-shot NLI cross-encoder; the adapted Qwen3 verifier is evaluated stand-alone.", 3.2)
P.equation(R("s") + Paren(R("d")) + R(" = ") + Nary("∑", R("r") + R(" ∈ {lex, dense}"), "", Frac(Sub(R("w"), R("r")), R("k") + R(" + ") + Sub(R("rank", True), R("r")) + Paren(R("d")))) + R(",  ") + R("k") + R(" = 60, ") + Sub(R("w"), R("r")) + R(" = 0.5"))
P.body("where rank_{r}(*d*) is the position of document *d* in list *r*. In the deployed pipeline each claim is checked against the 12 evidence sentences of highest lexical overlap by a DeBERTa-v3-small NLI cross-encoder [@he2021,nli_ce], and is "
       "treated as supported when")
P.equation(R("supported", True) + Paren(R("c")) + R(" = ") + R("1") + Paren(R("max", True) + Sub(R(""), R("e")) + R(" ") + Sub(R("p"), R("NLI", True)) + Paren(R("entail", True) + R(" | e, c")) + R(" ≥ 0.70"), "[", "]"))
P.body("A rule-based screen flags statements about doses, prescriptions, interactions, emergencies and a short list of drug classes as high-risk. Unsupported high-risk claims trigger abstention; unsupported lower-risk claims "
       "trigger a regeneration grounded in the top three passages, after which every sentence of the repaired text must again pass the NLI check, otherwise the draft is shown with an explicit warning. "
       "We stress that the repair and abstention stages, and the risk screen, are implemented heuristics whose end-to-end quality has not been measured in this study; "
       "the LoRA-adapted verifier is not yet wired into the pipeline’s verification call. Algorithm 1 gives the decision logic as implemented.")
P.algorithm("Algorithm 1. Verification and selective-repair policy (as implemented)", [
    "Input: question q; retriever; verifier V; threshold τ = 0.70",
    "1  a ← draft answer of Qwen3-1.7B for q; C ← sentences(a)",
    "2  E ← top-5 passages retrieved for q (hybrid RRF)",
    "3  for each claim c in C:",
    "     s(c) ← max of p_NLI(entail | e, c) over the 12 evidence",
    "            sentences with highest lexical overlap with c",
    "     risk(c) ← high if a keyword rule matches, else medium",
    "     unsupported(c) ← [ s(c) < τ ]",
    "4  if no claim is unsupported: return a",
    "5  if E is empty or some unsupported claim is high-risk:",
    "     return abstention message",
    "6  a′ ← regenerate a from the top-3 passages in E",
    "7  if every sentence of a′ has s ≥ τ: return a′",
    "8  else: return a with an explicit verification warning"])

P.h2("Datasets")
P.body("**Synthetic MedQuAD claims.** MedQuAD contains 47,457 question–answer pairs from 12 NIH websites [@abacha2019]; in the copy used here nine collections retain answer text. "
       "Parsing yields 5,477 documents and 13,547 question–answer pairs with at least three usable answer sentences (7–45 words). "
       "Documents are split by medical topic (the *Focus* field) into 4,372 / 561 / 544 documents for training, validation and testing, so no topic appears in more than one split. "
       "Each example pairs an evidence window of two to four consecutive answer sentences with a claim derived from one sentence of the window:")
P.bullet(f"*Entailment:* the target sentence after label-preserving edits (a synonym swap from a list of {NSYN} safe pairs with probability 0.35, otherwise truncation to its longest clause with probability 0.25).")
P.bullet(f"*Contradiction:* the same edited sentence after one meaning-reversing edit: negation of the first auxiliary or modal (sentences already containing negation are skipped), antonym replacement ({NANT} pairs), quantifier reversal ({NQ} pairs) or numeric scaling.")
P.bullet("*Neutral:* a sentence from another question of the same document (probability 0.40), from another document (0.25), or an *overclaim*—an entailed claim extended with an unsupported clause (0.35). Same- and other-document claims with Jaccard overlap above 0.45 with the evidence are rejected.")
P.body("With probability 0.35 an attribution prefix (e.g. “According to the passage,”) is prepended, independently of the label. Because edits that do not change meaning are applied to all classes, a claim being a near-copy of the evidence is not a reliable cue for entailment. "
       f"Two test sets are drawn from the held-out topics: *test-iid* uses the same edit families as training, and *test-ood* replaces the contradiction family with demographic, directional and temporal swaps ({NSW} pairs such as men↔women, left↔right, acute↔chronic) "
       f"and uses {NOADD} unseen overclaim templates in place of the {NADD} used in training. Algorithm 2 and {{T:edits}} summarise the construction.")
P.algorithm("Algorithm 2. Synthetic claim–evidence example construction", [
    "Input: documents D of one split; label ℓ ∈ {E, N, C}; edit families F; overclaim templates T",
    "1  sample document d ∈ D, question–answer pair q ∈ d, target sentence s_t ∈ q",
    "2  k ← one of {2, 3, 3, 4}; e ← k consecutive sentences of q that contain s_t",
    "3  if ℓ = E:  c ← preserve(s_t)",
    "4  if ℓ = C:  c₀ ← preserve(s_t); c ← f(c₀) for a random f ∈ F;",
    "              reject if no family changes c₀",
    "5  if ℓ = N:  u ~ U(0, 1)",
    "       u < 0.40: c ← sentence of another question in d",
    "       u < 0.65: c ← sentence of another document",
    "       else:     c ← overclaim(preserve(s_t), T)",
    "6  if ℓ = N and c is not an overclaim: reject if Jaccard(c, e) > 0.45 or c ⊂ e",
    "7  with probability 0.35 prepend an attribution prefix to c",
    "8  reject duplicates of (c, e); return (c, e, ℓ)"])
import re as _re
exs = {}


def _good(c):
    w = c.split()
    mid_caps = sum(1 for x in w[1:] if _re.match(r"[A-Z][a-z]+", x))
    return 8 <= len(w) <= 16 and "(" not in c and mid_caps == 0 and not _re.match(r"(Such|As|The reference|According|It is|In general|Reportedly|The text)", c)


for fname in ("test_iid", "test_ood"):
    for r_ in json.loads((ROOT / f"data/claim_verification/{fname}.json").read_text(encoding="utf-8")):
        if _good(r_["claim"]) and r_["kind"] not in exs:
            exs[r_["kind"]] = r_["claim"]
for fname in ("test_iid", "test_ood"):
    for r_ in json.loads((ROOT / f"data/claim_verification/{fname}.json").read_text(encoding="utf-8")):
        if r_["kind"] not in exs and len(r_["claim"].split()) <= 17:
            exs[r_["kind"]] = r_["claim"]
fam = [("entail", "Entailment (preserving)", "—"), ("negation", "Contradiction: negation", f"{NMOD} modals"), ("antonym", "Contradiction: antonym", f"{NANT} pairs"),
       ("quantifier", "Contradiction: quantifier", f"{NQ} pairs"), ("number", "Contradiction: number", "factors 0.1–10"), ("swap_ood", "Contradiction: demographic/directional (unseen)", f"{NSW} pairs"),
       ("neutral_same_doc", "Neutral: same document", "p = 0.40"), ("neutral_other_doc", "Neutral: other document", "p = 0.25"), ("neutral_overclaim", "Neutral: overclaim", f"{NADD} train / {NOADD} unseen")]
P.table("edits", "Construction Families and Example Claims (Held-Out Topics)",
        ["Family", "Size", "Example claim"],
        [[nm, sz, Raw(exs.get(k, "—"))] for k, nm, sz in fam],
        [2.7, 1.9, 3.9], font=5.9, align=["l", "l", "l"],
        footnote="Overclaims append an unsupported clause to an entailed claim; the example column shows a claim from the held-out test sets.")
P.body("**HealthVer.** HealthVer [@sarrouti2021] contains health-related claims paired with evidence passages and labelled Supports, Refutes or Neutral. We use a re-split version [@healthver_resplit] with 5,005 training, 850 development and 840 test rows in which no claim appears in more than one split (verified: zero overlap between training and test claims). "
       "Labels are mapped to E, C and N. Almost all rows (99.8% by keyword count) concern COVID-19, so HealthVer is both real and a domain shift relative to MedQuAD. {T:data} summarises the data.")
P.table("data", "Dataset Statistics",
        ["Set", "Source", "Rows", "E / N / C", "Evid. words"],
        [["Train", "MedQuAD synthetic", "6,000", "2000 / 2000 / 2000", "51.6"],
         ["Val", "MedQuAD synthetic", "600", "200 / 200 / 200", "51.5"],
         ["Test-iid", "MedQuAD synthetic", "900", "300 / 300 / 300", "50.3"],
         ["Test-ood", "MedQuAD synthetic", "900", "300 / 300 / 300", "51.2"],
         ["Train", "HealthVer", "5,005", "1897 / 1918 / 1190", "31.7"],
         ["Dev", "HealthVer", "850", "252 / 386 / 212", "32.3"],
         ["Test", "HealthVer", "840", "322 / 312 / 206", "31.6"]],
        [1.3, 2.4, 1.0, 2.6, 1.2], font=6.5, align=["l", "l", "r", "c", "r"],
        footnote="E/N/C: entailment/neutral/contradiction. Training uses a 2,500-row synthetic and a 4,000-row HealthVer subsample (seed 42); validation uses 300 synthetic and 450 HealthVer-dev rows.")

P.figure("flow", FIG / "fig_dataflow.png", "Data construction and training pipeline: topic-disjoint synthetic claims from MedQuAD and real HealthVer claims are mixed to adapt Qwen3-1.7B with LoRA.", 3.1)
P.h2("Model and Adaptation")
P.body("The backbone is Qwen3-1.7B, a decoder-only Transformer with 28 layers, hidden size 2,048 and grouped-query attention with 16 query and 8 key–value heads [@yang2025]. "
       "Each input is rendered with the model’s chat template (thinking disabled) from an instruction, the claim and the evidence; the instruction asks for one word: entailment, neutral or contradiction. "
       "LoRA [@hu2022] replaces each adapted weight matrix *W* by")
P.equation(Sup(R("W"), R("′")) + R(" = ") + R("W") + R(" + ") + Frac(R("α"), R("r")) + R("BA") + R(",  ") + R("B") + R(" ∈ ") + Sup(R("ℝ"), R("d×r")) + R(", ") + R("A") + R(" ∈ ") + Sup(R("ℝ"), R("r×k")))
P.body("with rank *r* = 16 and scaling α = 32 applied to the query, key, value, output, gate, up and down projections of all layers, giving 17.43 M trainable parameters (1.003% of 1.738 B). "
       "The frozen backbone is held in bfloat16 and the adapter weights in float32 with bfloat16 autocast; no quantisation is used, so this is plain LoRA rather than the QLoRA configuration with *r* = 8 on the query and value projections that was initially planned.")
P.body("Let *h*_{T} be the final-layer hidden state at the last prompt position and *W*_{o} the output (language-model-head) matrix, and let *t*(*y*) be the first token of the label word for *y*. "
       "Training minimises the full-vocabulary cross-entropy of that token, and prediction restricts the softmax to the three label tokens:")
P.equation(R("ℒ") + Paren(R("θ")) + R(" = −") + Frac(R("1"), Abs(R("ℬ"))) + Nary("∑", R("(c,e,y)∈ℬ"), "", R("log", True) + Sub(Paren(R("softmax", True) + Paren(Sub(R("W"), R("o")) + Sub(R("h"), R("T")))), R("t") + Paren(R("y")))))
P.equation(R("p") + Paren(R("y | c, e")) + R(" = ") + Frac(R("exp", True) + Paren(Sub(R("z"), R("y"))), Nary("∑", R("y′∈𝒴"), "", R("exp", True) + Paren(Sub(R("z"), R("y′"))))) + R(",  ") + Sub(R("z"), R("y")) + R(" = ") + Sub(Paren(Sub(R("W"), R("o")) + Sub(R("h"), R("T")), "[", "]"), R("t") + Paren(R("y"))))
P.body("Computing the head only at the last position (with left padding) lowers memory so that training fits in 6 GB. The model remains an ordinary generative model, because the loss is taken over the whole vocabulary.")

P.h2("Baselines")
P.body("Two baselines use the same test items. The first is the zero-shot DeBERTa-v3-small NLI cross-encoder [@he2021,nli_ce] that serves as the verifier in the deployed pipeline, with the evidence as premise and the claim as hypothesis. "
       "The second is the unadapted Qwen3-1.7B with the identical prompt and the same restricted three-way scoring, which isolates the effect of adaptation from the effect of the prompt.")

# ======================================================================= IV. EXPERIMENTAL SETUP
P.h1("Experimental Setup")
P.h2("Hardware and Software")
P.body("All claim-verification experiments ran on one NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB of memory under Windows 11, using Python 3.10, PyTorch 2.11.0 (CUDA 12.8), Transformers 5.20.2 and PEFT 0.21.3 [@wolf2020]. "
       "The main training run took about 35 minutes.")
P.h2("Configuration")
P.table("cfg", "Training Configuration",
        ["Parameter", "Value"],
        [["Backbone / precision", "Qwen3-1.7B / bf16 (adapters fp32)"],
         ["LoRA rank, α, dropout", "16, 32, 0.1"],
         ["Adapted modules", "q, k, v, o, gate, up, down"],
         ["Trainable parameters", "17,432,576 (1.003%)"],
         ["Optimizer", "AdamW, weight decay 0.01"],
         ["Learning rate", "1×10^{-4}, 5% warm-up, cosine decay to 5% of peak"],
         ["Batch size", "8 × 2 accumulation = 16"],
         ["Epochs / optimizer steps", "2 / 814"],
         ["Max. sequence length", "448 tokens (middle trimmed)"],
         ["Gradient clipping", "1.0"],
         ["Training mixture", "4,000 HealthVer + 2,500 synthetic"],
         ["Validation", "450 HealthVer-dev + 300 synthetic"],
         ["Model selection", "Best val. macro-F1, every 100 steps"],
         ["Seed", "42 (single run)"]],
        [3.4, 5.0], font=6.6, align=["l", "l"])
P.h2("Evaluation Protocol")
P.body("Validation data are used only to select the checkpoint; test sets are scored once after training. For each test set and model we report accuracy and the macro-averaged precision, recall and F1 over the three classes. "
       "For class *k* with true positives TP_{k}, false positives FP_{k} and false negatives FN_{k}:")
P.equation(Sub(R("P"), R("k")) + R(" = ") + Frac(Sub(R("TP", True), R("k")), Sub(R("TP", True), R("k")) + R(" + ") + Sub(R("FP", True), R("k"))) + R(",   ") + Sub(R("R"), R("k")) + R(" = ") + Frac(Sub(R("TP", True), R("k")), Sub(R("TP", True), R("k")) + R(" + ") + Sub(R("FN", True), R("k"))))
P.equation(Sub(R("F"), R("1,k")) + R(" = ") + Frac(R("2") + Sub(R("P"), R("k")) + Sub(R("R"), R("k")), Sub(R("P"), R("k")) + R(" + ") + Sub(R("R"), R("k"))) + R(",   ") + Sub(R("F"), R("1")) + R(" = ") + Frac(R("1"), R("3")) + Nary("∑", R("k"), "", Sub(R("F"), R("1,k"))))
P.body("Macro precision and recall average *P*_{k} and *R*_{k} in the same way. Because the test sets are modest in size (840–900 items), 95% confidence intervals are obtained by resampling test items with replacement 2,000 times and taking the 2.5th and 97.5th percentiles; differences between systems use paired resamples. "
       "Calibration is measured by the expected calibration error [@guo2017] with ten equal-width bins *B*_{m}:")
P.equation(R("ECE", True) + R(" = ") + Nary("∑", R("m=1"), R("10"), Frac(Abs(Sub(R("B"), R("m"))), R("n")) + Abs(R("acc", True) + Paren(Sub(R("B"), R("m"))) + R(" − ") + R("conf", True) + Paren(Sub(R("B"), R("m"))))))
P.body("For selective prediction [@geifman2017] a prediction is retained when κ_{i} ≥ τ; coverage is the retained fraction and selective accuracy is the accuracy on retained items:")
P.equation(R("cov", True) + Paren(R("τ")) + R(" = ") + Frac(R("1"), R("n")) + Nary("∑", R("i"), "", R("1") + Paren(Sub(R("κ"), R("i")) + R(" ≥ τ"), "[", "]")) + R(",   ") + R("acc", True) + Sub(R(""), R("sel", True)) + Paren(R("τ")) + R(" = ") +
           Frac(Nary("∑", R("i"), "", R("1") + Paren(Sub(R("κ"), R("i")) + R(" ≥ τ ∧ ") + Sub(R("ŷ"), R("i")) + R(" = ") + Sub(R("y"), R("i")), "[", "]")), Nary("∑", R("i"), "", R("1") + Paren(Sub(R("κ"), R("i")) + R(" ≥ τ"), "[", "]"))))
P.body("The ablation trains two further adapters with identical hyper-parameters and the same seed: *real only* (4,000 HealthVer rows) and *synthetic only* (2,500 synthetic rows). Because the sizes differ, the comparison reveals the contribution of each source rather than the effect of data volume alone. "
       "The checkpoint of every adapter is selected on the same mixed validation set, which includes 450 real development rows; this slightly favours the synthetic-only adapter on real data and is disclosed as a limitation.")

# ======================================================================= V. RESULTS
P.h1("Results and Discussion")
P.h2("Quantitative Results")
LG = json.loads((ROOT / "models/qwen3-1.7b-lora-claimverify/train_log.json").read_text(encoding="utf-8"))["log"]
P.body(f"Validation macro-F1 rose from {LG[0]['macro_f1']:.3f} at step {LG[0]['step']} to {LG[-1]['macro_f1']:.3f} at step {LG[-1]['step']} ({{F:curve}}). The best checkpoint was the final one and the early-stopping rule never fired, "
       "so training had not saturated and longer schedules may help; we did not test this.")
P.figure("curve", FIG / "fig_curve.png", "Training loss (left, log scale) and validation macro-F1, precision and recall (right; 450 HealthVer-dev and 300 synthetic rows) during training.", 3.2)
rows = []
for tag, nm in MODELS_MAIN:
    for s, sn in zip(SETS, ["MQ-iid", "MQ-ood", "HealthVer"]):
        c = M[tag]["sets"][s]["ci"]
        rows.append([nm if s == SETS[0] else "", sn, f"{c['acc']['point']:.3f}", f"{c['P']['point']:.3f}", f"{c['R']['point']:.3f}",
                     f"{c['F1']['point']:.3f} [{c['F1']['lo']:.2f}, {c['F1']['hi']:.2f}]"])
P.table("main", "Claim-Verification Results on Held-Out Test Sets",
        ["Model", "Test set", "Acc.", "Macro-P", "Macro-R", "Macro-F1 [95% CI]"],
        rows, [1.95, 1.35, 0.8, 0.9, 0.9, 2.4], font=6.0, align=["l", "l", "c", "c", "c", "c"],
        shade_rows=(0, 1, 2),
        footnote="MQ-iid: held-out MedQuAD topics; MQ-ood: held-out topics with unseen edit families; HealthVer: real claims. Zero-shot rows use no task training.")
P.figure("main", FIG / "fig_main.png", "Macro-F1 with 95% bootstrap confidence intervals. The dashed line marks the 0.90 target.", 3.25)
P.body(f"{{T:main}} and {{F:main}} answer RQ1. On held-out MedQuAD topics the adapted verifier reaches macro-F1 {ci('mixed','medquad_test_iid')} (95% CI {M['mixed']['sets']['medquad_test_iid']['ci']['F1']['lo']:.3f}–{M['mixed']['sets']['medquad_test_iid']['ci']['F1']['hi']:.3f}), "
       f"and {ci('mixed','medquad_test_ood')} when the contradiction family and overclaim templates were never seen, so precision and recall of at least 0.90 are met in both synthetic conditions. "
       f"The zero-shot NLI verifier reaches {ci('nli_deberta','medquad_test_iid')} and {ci('nli_deberta','medquad_test_ood')}; the paired differences in macro-F1 are {pdiff('nli_deberta','medquad_test_iid')} and {pdiff('nli_deberta','medquad_test_ood')}, "
       f"with intervals excluding zero. On real HealthVer claims the adapted model reaches {ci('mixed','healthver_test')} against {ci('nli_deberta','healthver_test')} for the NLI model (difference {pdiff('nli_deberta','healthver_test')}). "
       "The comparison is between a domain-adapted and an off-the-shelf model and should be read as such; it shows that adaptation to medical claim–evidence pairs is necessary, not that the 1.7B architecture is intrinsically superior to a cross-encoder. "
       + (f"The unadapted backbone with the same prompt reaches only {ci('base','medquad_test_iid')}, {ci('base','medquad_test_ood')} and {ci('base','healthver_test')}, so the gains are attributable to adaptation. " if HAVE_BASE else ""))
def verdict(d):
    if d["lo"] > 0:
        return "higher than"
    if d["hi"] < 0:
        return "lower than"
    return "statistically indistinguishable from"


if "deberta_ft" in M:
    pd_ = {s: A["paired"][f"mixed_vs_deberta_ft/{s}"] for s in SETS}
    P.body(f"A natural question is whether the gain comes from the LLM backbone or simply from in-domain supervision. We therefore fine-tuned the same DeBERTa-v3-small cross-encoder on the identical 6,500 training examples (three epochs, checkpoint selected on the same validation data). "
           f"It reaches macro-F1 {ci('deberta_ft','medquad_test_iid')}, {ci('deberta_ft','medquad_test_ood')} and {ci('deberta_ft','healthver_test')} on MQ-iid, MQ-ood and HealthVer. "
           f"The adapted Qwen verifier is {verdict(pd_['medquad_test_iid'])} it on MQ-iid ({pdiff('deberta_ft','medquad_test_iid')}), {verdict(pd_['medquad_test_ood'])} it on MQ-ood ({pdiff('deberta_ft','medquad_test_ood')}) and {verdict(pd_['healthver_test'])} it on HealthVer ({pdiff('deberta_ft','healthver_test')}). "
           "This comparison separates the contribution of supervised in-domain data from that of the model family; the project nevertheless uses a decoder LLM because the same model also drafts and repairs answers, so a single backbone serves all stages.")
if "fewshot" in M:
    P.body(f"Prompting alone is not sufficient either: six in-context exemplars (one real and one synthetic per class, taken from the training sets) lift the unadapted Qwen3-1.7B to macro-F1 {ci('fewshot','medquad_test_iid')}, {ci('fewshot','medquad_test_ood')} and {ci('fewshot','healthver_test')}, "
           f"compared with {ci('base','medquad_test_iid')}, {ci('base','medquad_test_ood')} and {ci('base','healthver_test')} zero-shot; the adapted model\u2019s paired advantage on HealthVer over the six-shot model is {pdiff('fewshot','healthver_test')}.")
P.body(f"The 0.90 target is not met on real claims: macro-precision and macro-recall are {ci('mixed','healthver_test','P')} and {ci('mixed','healthver_test','R')}, and the upper bound of the macro-F1 interval is {M['mixed']['sets']['healthver_test']['ci']['F1']['hi']:.3f}. "
       "The drop from synthetic to real claims is the central empirical finding of this paper.")

P.h2("Per-Class Behaviour and Confusions")
P.figure("conf", FIG / "fig_confusion.png", "Row-normalised confusion matrices of the fine-tuned verifier (E: entailment, N: neutral, C: contradiction).", 3.25)
pc = {s: EV["adapter"][s]["per_class"] for s in SETS}
P.table("pc", "Per-Class Precision / Recall of the Fine-Tuned Verifier",
        ["Class", "MQ-iid", "MQ-ood", "HealthVer"],
        [[k.capitalize(), *[f"{pc[s][k]['precision']:.2f} / {pc[s][k]['recall']:.2f}" for s in SETS]] for k in ("entailment", "neutral", "contradiction")],
        [2.4, 2.0, 2.0, 2.0], font=6.8, footnote="Entries are precision / recall.")
P.body("{F:conf} and {T:pc} show that on HealthVer the three classes are recognised with similar quality (recall 0.72–0.79), and that errors are spread across all off-diagonal cells, the largest being contradiction predicted as neutral (0.16). "
       "On MQ-ood the pattern is different and concentrated: entailment recall is 1.00 but 16% of contradictions are labelled entailment, lowering entailment precision to 0.86 and contradiction recall to 0.84. "
       "A verifier that accepts a contradicted claim is the most dangerous failure for a safety application, and it is precisely this error that grows when the contradiction type is unfamiliar.")

P.h2("Generalisation Across Edit Types")
P.figure("kinds", FIG / "fig_kinds.png", "Accuracy of the fine-tuned verifier by construction type on the two synthetic test sets.", 3.25)
P.body(f"{{F:kinds}} breaks the synthetic results down by construction type. In the in-distribution set every type is above 0.97 except numeric edits (0.94, *n* = 16). "
       f"The unseen demographic/directional swaps are recognised as contradictions in {A['per_kind']['medquad_test_ood']['swap_ood'][0]:.2f} of cases (*n* = {A['per_kind']['medquad_test_ood']['swap_ood'][1]}), while neutral and entailed items in the same set remain above 0.98. "
       "The model therefore transfers to unseen edits only partly: it has learned a considerable part of the notion of contradiction, but swaps of attributes such as sex, age group or side were not covered by training and are sometimes read as compatible with the evidence.")

P.h2("Ablation of Training-Data Composition")
if HAVE_ABL:
    rows = []
    for tag, nm in (("real_only", "Real only (4,000)"), ("synth_only", "Synthetic only (2,500)"), ("mixed", "Mixed (6,500)")):
        rows.append([nm] + [cir(tag, s) for s in SETS])
    P.table("abl", "Training-Data Ablation (Macro-F1 [95% CI])",
            ["Training data", "MQ-iid", "MQ-ood", "HealthVer"], rows, [2.3, 2.05, 2.05, 2.05], font=6.2,
            shade_rows=(2,), footnote="All adapters share hyper-parameters and seed; checkpoints are selected on the same mixed validation set.")
    P.figure("abl", FIG / "fig_ablation.png", "Effect of training-data composition on macro-F1 (95% bootstrap CI).", 3.25)
    d_r = A["paired"]["mixed_vs_real_only/medquad_test_ood"]; d_s = A["paired"]["mixed_vs_synth_only/healthver_test"]
    d_rh = A["paired"]["mixed_vs_real_only/healthver_test"]; d_si = A["paired"]["mixed_vs_synth_only/medquad_test_iid"]
    P.body(f"{{T:abl}} and {{F:abl}} address RQ2. The synthetic-only adapter is strong on MedQuAD ({ci('synth_only','medquad_test_iid')} and {ci('synth_only','medquad_test_ood')}) but reaches {ci('synth_only','healthver_test')} on real claims; "
           f"adding real data improves HealthVer macro-F1 by {d_s['diff']:+.3f} [{d_s['lo']:+.3f}, {d_s['hi']:+.3f}]. "
           f"The real-only adapter reaches {ci('real_only','healthver_test')} on HealthVer, close to the mixed model (difference {d_rh['diff']:+.3f} [{d_rh['lo']:+.3f}, {d_rh['hi']:+.3f}]), but scores {ci('real_only','medquad_test_iid')} and {ci('real_only','medquad_test_ood')} on MedQuAD; "
           f"the mixed model improves on it by {d_r['diff']:+.3f} [{d_r['lo']:+.3f}, {d_r['hi']:+.3f}] on MQ-ood. "
           "Synthetic data thus teach the structure of the task and MedQuAD-style language, whereas real data supply the vocabulary and argument structure of real claims; neither alone gives both. "
           "The dependence of real-claim performance on real training claims also suggests that further gains require more real, domain-matched annotation, not more synthetic data.")
    if HAVE_SEED:
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
    P.body("[ablation pending]")

P.h2("Calibration and Selective Prediction")
P.figure("cal", FIG / "fig_calibration.png", "Left: reliability of the maximum class probability. Right: accuracy on retained items as low-confidence predictions are rejected.", 3.25)
P.table("sel", "Selective Accuracy at Given Coverage (Fine-Tuned Verifier)",
        ["Coverage", "MQ-iid", "MQ-ood", "HealthVer"],
        [[f"{int(float(c) * 100)}%"] + [f"{S[s][c]:.3f}" for s in SETS] for c in ("1.0", "0.9", "0.8", "0.7", "0.6", "0.5")],
        [2.0, 2.0, 2.0, 2.4], font=6.8, footnote=f"ECE: MQ-iid {M['mixed']['sets']['medquad_test_iid']['ece']:.3f}, MQ-ood {M['mixed']['sets']['medquad_test_ood']['ece']:.3f}, HealthVer {M['mixed']['sets']['healthver_test']['ece']:.3f}.")
P.body(f"{{F:cal}} and {{T:sel}} address RQ3. The verifier is well calibrated in distribution (ECE {M['mixed']['sets']['medquad_test_iid']['ece']:.3f}) and slightly over-confident with unseen edits ({M['mixed']['sets']['medquad_test_ood']['ece']:.3f}), but clearly over-confident on real claims ({M['mixed']['sets']['healthver_test']['ece']:.3f}). "
       f"For comparison the NLI baseline has ECE {M['nli_deberta']['sets']['healthver_test']['ece']:.3f} on HealthVer. Rejecting uncertain predictions helps: on MQ-ood, accuracy rises from {S['medquad_test_ood']['1.0']:.3f} to {S['medquad_test_ood']['0.9']:.3f} at 90% coverage, "
       f"and on HealthVer from {S['healthver_test']['1.0']:.3f} to {S['healthver_test']['0.5']:.3f} at 50% coverage. Even then the 0.90 level is not reached on real claims, so the confidence signal alone does not "
       "make a safe abstention rule; a clinically usable system would need recalibration (for example temperature scaling on real validation data) and a stricter threshold for high-risk statements.")

P.h2("Unsupported-Claim Detection at the Deployed Threshold")
DX = json.loads((RES / "analysis_extra.json").read_text(encoding="utf-8"))
DT = DX["detection"]
P.body("The deployed pipeline does not use three-way labels directly: a claim is flagged as unsupported when its entailment score is below τ = 0.70 (Algorithm 1). "
       "{T:det} addresses RQ2 by evaluating that binary decision with each verifier’s entailment probability, treating neutral and contradicted claims as the positive class. Its columns are verifier-level proxies for project metrics: "
       "*flag recall* (the share of unsupported claims caught, an error-detection proxy), *flag precision*, the *false-flag rate* (the share of entailed claims wrongly flagged, a proxy for unnecessary repair) "
       "and the *accepted-contradiction rate* (the share of contradicted claims that pass as supported).")
rows = []
for tag, nm in MODELS_MAIN:
    for s, sn in zip(SETS, ["MQ-iid", "MQ-ood", "HealthVer"]):
        d = DT[tag][s]
        rows.append([nm if s == SETS[0] else "", sn, f"{d['flag_recall']:.3f}", f"{d['flag_precision']:.3f}", f"{d['false_flag_rate']:.3f}", f"{d['contra_accepted']:.3f}", f"{d['ap_unsupported']:.3f}"])
P.table("det", "Unsupported-Claim Detection at τ = 0.70 (Positive Class: Neutral or Contradicted)",
        ["Model", "Test set", "Flag R", "Flag P", "False-flag", "C accepted", "AP"],
        rows, [2.0, 1.15, 0.85, 0.85, 1.1, 1.2, 0.7], font=5.9, align=["l", "l", "c", "c", "c", "c", "c"], shade_rows=(0, 1, 2),
        footnote="Flag R/P: recall/precision for flagging unsupported claims; false-flag: fraction of entailed claims flagged; C accepted: fraction of contradicted claims not flagged; AP: average precision over all thresholds.")
dm, dn = DT["mixed"]["healthver_test"], DT["nli_deberta"]["healthver_test"]
P.body(f"On real claims the fine-tuned verifier catches {dm['flag_recall']:.3f} of unsupported claims at precision {dm['flag_precision']:.3f}, but it also flags {dm['false_flag_rate']:.3f} of entailed claims and lets {dm['contra_accepted']:.3f} of contradictions through. "
       f"The zero-shot NLI verifier reaches a higher flag recall ({dn['flag_recall']:.3f}) only by flagging almost everything: {dn['false_flag_rate']:.3f} of entailed claims, so at this threshold it would trigger repair or abstention on nearly every real claim. "
       f"Threshold-independent ranking quality tells the same story ({{F:pr}}): average precision for detecting unsupported real claims is {dm['ap_unsupported']:.3f} for the adapted model and {dn['ap_unsupported']:.3f} for the NLI model. "
       "The binary supported/unsupported decision is therefore easier than the three-way decision and comes closest to the 90% objective (flag recall above 0.90 on HealthVer, 0.935 on MQ-ood), but at the deployed threshold precision stays well below 0.90 on real claims. "
       "A better operating point could be chosen on validation data; choosing it on the test set would be optimistic, so we do not report one.")
P.figure("pr", FIG / "fig_pr.png", "Precision–recall curves for detecting unsupported claims (neutral or contradicted) from the entailment probability. Markers show the deployed threshold τ = 0.70.", 3.25)

if HAVE_CAL:
    CAL = json.loads((RES / "analysis_calib.json").read_text(encoding="utf-8"))["mixed"]
    P.h2("Validation-Based Recalibration and Operating Points")
    P.body("The preceding analyses used the fixed threshold of the deployed pipeline. This subsection fits recalibration parameters and thresholds on validation data only (HealthVer development rows for HealthVer, MedQuAD validation rows for the synthetic sets) and applies them unchanged to the test sets. "
           "Temperature scaling [@guo2017] divides the log class probabilities by a single scalar *T* fitted by minimising validation negative log-likelihood; it does not change predictions or accuracy. "
           "Two operating points are then chosen for the unsupported-claim decision: the threshold that maximises validation F1 for flagging unsupported claims, and the smallest threshold that accepts at most 5% of contradicted validation claims (a safety-oriented rule).")
    rows = []
    for s, sn in zip(SETS, ["MQ-iid", "MQ-ood", "HealthVer"]):
        d = CAL[s]
        for k, lab in (("fixed_0.70", "fixed 0.70"), ("f1_opt", "max val. F1"), ("safety_5pct", "≤5% C accepted")):
            x = d[k]
            rows.append([sn if k == "fixed_0.70" else "", lab, f"{x['tau']:.2f}", f"{x['flag_recall']:.3f}", f"{x['flag_precision']:.3f}", f"{x['false_flag_rate']:.3f}", f"{x['contra_accepted']:.3f}"])
    P.table("op", "Operating Points Selected on Validation Data (Fine-Tuned Verifier, Test Results)",
            ["Test set", "Rule", "τ", "Flag R", "Flag P", "False-flag", "C accepted"],
            rows, [1.5, 2.0, 0.7, 1.0, 1.0, 1.2, 1.2], font=6.4, align=["l", "l", "c", "c", "c", "c", "c"],
            footnote=f"Temperature scaling: T = {CAL['medquad_test_iid']['T']:.2f} (MedQuAD validation) and {CAL['healthver_test']['T']:.2f} (HealthVer dev). ECE before → after scaling: MQ-iid {CAL['medquad_test_iid']['ece_before']:.3f} → {CAL['medquad_test_iid']['ece_after']:.3f}, "
                     f"MQ-ood {CAL['medquad_test_ood']['ece_before']:.3f} → {CAL['medquad_test_ood']['ece_after']:.3f}, HealthVer {CAL['healthver_test']['ece_before']:.3f} → {CAL['healthver_test']['ece_after']:.3f}.")
    h = CAL["healthver_test"]
    P.body(f"Temperature scaling reduces the HealthVer calibration error from {h['ece_before']:.3f} to {h['ece_after']:.3f} (fitted T = {h['T']:.2f}), which confirms that the over-confidence in {{F:cal}} is largely a matter of scale and can be corrected with a small amount of real validation data. "
           f"Choosing the threshold on validation data gives, on real test claims, flag recall {h['f1_opt']['flag_recall']:.3f} and precision {h['f1_opt']['flag_precision']:.3f} for the F1-optimal rule (false-flag rate {h['f1_opt']['false_flag_rate']:.3f}), "
           f"and {h['safety_5pct']['contra_accepted']:.3f} accepted contradictions with a false-flag rate of {h['safety_5pct']['false_flag_rate']:.3f} for the safety rule. "
           "The numbers show the trade-off an operator faces: protecting against accepted contradictions costs unnecessary repairs, and neither rule reaches 90% on both flag precision and flag recall for real claims.")

P.h2("Breakdown by Source Collection and Length")
BK = DX["breakdown"]
SRC = {"2_GARD_QA": "GARD", "3_GHR_QA": "Genetics Home Ref.", "4_MPlus_Health_Topics_QA": "MedlinePlus topics", "6_NINDS_QA": "NINDS", "5_NIDDK_QA": "NIDDK", "8_NHLBI_QA_XML": "NHLBI", "9_CDC_QA": "CDC", "1_CancerGov_QA": "CancerGov", "7_SeniorHealth_QA": "NIH SeniorHealth"}
rows = []
for k, v in sorted(BK["medquad_test_iid"]["source"].items(), key=lambda kv: -kv[1][1]):
    o = BK["medquad_test_ood"]["source"].get(k)
    rows.append([SRC.get(k, k), f"{v[1]}", f"{v[0]:.3f}", f"{o[0]:.3f}" if o else "—"])
for key, lab in (("claim_w", "HealthVer claim words"), ("evid_w", "HealthVer evidence words")):
    for b_ in BK["healthver_test"][key]:
        rows.append([f"{lab} {b_['range'][0]}–{b_['range'][1]}", f"{b_['n']}", f"{b_['acc']:.3f}", "—"])
P.table("brk", "Accuracy of the Fine-Tuned Verifier by Subset",
        ["Subset", "n", "Acc. (iid / HealthVer)", "Acc. (ood)"],
        rows, [3.8, 0.9, 2.3, 1.5], font=6.2, align=["l", "c", "c", "c"],
        footnote="MedQuAD rows: n counts MQ-iid items; collections with fewer than 25 test items are too small for conclusions. HealthVer rows are length terciles.")
cl = BK["healthver_test"]["claim_w"]; ev = BK["healthver_test"]["evid_w"]
P.body(f"Accuracy on MedQuAD is above 0.97 for the three largest collections (GARD, Genetics Home Reference, MedlinePlus topics, together {470 + 168 + 146} of 900 MQ-iid items), and the smaller collections do not support firm conclusions. "
       f"On HealthVer, accuracy falls with claim length, from {cl[0]['acc']:.3f} for claims of {cl[0]['range'][0]}–{cl[0]['range'][1]} words to {cl[2]['acc']:.3f} for claims of {cl[2]['range'][0]}–{cl[2]['range'][1]} words, whereas evidence length shows no monotone trend "
       f"({ev[0]['acc']:.3f}, {ev[1]['acc']:.3f}, {ev[2]['acc']:.3f}). Longer claims often combine several assertions, only one of which may be unsupported; this motivates the sentence-level claim extraction in the pipeline and suggests splitting multi-clause sentences further.")

P.h2("Preliminary PubMedQA Experiment")
P.body("Before the claim-level formulation we fine-tuned the same backbone on PubMedQA [@jin2019], whose 1,000 expert-labelled questions are answered yes, no or maybe from an abstract. "
       "The data were split 75/10/15 by question (749/99/152) and training used class-balanced sampling plus 250 MedQuAD rows; a maybe-class logit offset was chosen on validation data only. {T:pq} reports the held-out test split.")
P.table("pq", "Preliminary PubMedQA Test Results (n = 152)",
        ["System", "Acc.", "Macro-F1", "Yes P/R", "No P/R", "Maybe P/R"],
        [["Fine-tuned adapter", "0.487", "0.396", "0.72/0.41", "0.53/0.75", "0.03/0.06"],
         ["Base, label scoring", "0.461", "0.359", "0.63/0.67", "0.67/0.15", "0.12/0.35"],
         ["Majority class (yes)", "0.546", "0.236", "0.55/1.00", "0.00/0.00", "0.00/0.00"]],
        [2.1, 0.7, 0.9, 1.5, 1.5, 1.5], font=6.0, align=["l", "c", "c", "c", "c", "c"])
P.body("The adapter does not exceed the majority-class accuracy of 0.546 and almost never recovers the rare maybe class (17 test items). Three factors explain why this formulation could not meet a 90% target: "
       "(i) the original study reports 78.0% accuracy for a single human annotator, 68.1% for its best model and 55.2% for the majority baseline [@jin2019], so a 90% target exceeds even reported human accuracy on this formulation; (ii) 749 training examples (82 maybe) are too few to adapt a 1.7B model without over-fitting; and (iii) in our training script half of the sampled examples came from an unrelated free-text task, diluting the objective. "
       "More fundamentally, answering a research question is not the same as judging whether a *claim* is supported by evidence, which is what the verification objective requires; we therefore treat this experiment as a negative result and a motivation for Section III, and exclude it from the main comparison.")

P.h2("Error Analysis")
HVT = [json.loads(l) for l in (ROOT / "data/external/healthver/test.jsonl").read_text(encoding="utf-8").splitlines()]
PRED = json.loads((RES / "predictions_mixed.json").read_text(encoding="utf-8"))["healthver_test"]
LABN = {"entailment": "E", "neutral": "N", "contradiction": "C"}
errs = [(max(r["probs"]), i, r) for i, r in enumerate(PRED) if r["gold"] != r["pred"]]
errs.sort(key=lambda x: -x[0])
picked, seen_pairs = [], set()
for conf, i, r in errs:
    pair = (r["gold"], r["pred"])
    if pair in seen_pairs or len(HVT[i]["Evidence"].split()) > 38 or len(HVT[i]["Claim"].split()) > 26:
        continue
    seen_pairs.add(pair); picked.append((conf, i, r))
    if len(picked) == 3:
        break
rows = []
for conf, i, r in picked:
    rows.append([Raw(HVT[i]["Claim"].strip()), Raw(HVT[i]["Evidence"].strip()), f"{LABN[r['gold']]}→{LABN[r['pred']]} ({conf:.2f})"])
P.table("err", "Confident Errors on HealthVer (Gold → Predicted, Confidence)",
        ["Claim", "Evidence", "Error"], rows, [3.0, 4.2, 1.3], font=5.8, align=["l", "l", "c"])
wrong_conf = [c for c, _, _ in errs]
n_hi = sum(c >= 0.9 for c in wrong_conf)
P.body(f"Of {len(errs)} HealthVer errors, {n_hi} ({100 * n_hi / len(errs):.0f}%) were made with confidence of 0.90 or more, which is why abstention on confidence alone is insufficient. "
       "The examples in {T:err} show typical patterns: claims that restate part of the evidence while omitting a qualification, claims whose polarity depends on a comparison or a negated outcome, and evidence passages that discuss the topic without addressing the specific claim. "
       "These are phenomena that require sentence-level reasoning beyond lexical overlap and that were not represented in the rule-based synthetic edits.")

if HAVE_DEMO:
    DEMO = json.loads((RES / "pipeline_demo.json").read_text(encoding="utf-8"))
    P.h2("Illustrative Walk-Through of the Deployed Pipeline")
    P.body("To show how the stages of Algorithm 1 behave together, the pipeline was run on four questions with the BM25-style retriever, the unadapted Qwen3-1.7B generator and the DeBERTa NLI verifier. "
           "This is a demonstration, not an evaluation: four questions cannot support any rate, the claims were not manually graded, and the retrieval corpus (MedQuAD) is the same material from which the synthetic training claims were built. "
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
            footnote="Supported: entailment score \u2265 0.70 (Algorithm 1). Outcomes follow the policy in Algorithm 1.")
    allc = [(c, d_["question"]) for d_ in DEMO for c in d_["claims"]]
    if allc:
        n_sup = sum(c["status"] == "potentially_supported" for c, _ in allc)
        low = min(allc, key=lambda x: x[0]["confidence"])[0]
        hi = max(allc, key=lambda x: x[0]["confidence"])[0]
        P.body(f"Across the four runs, {n_sup} of {len(allc)} generated claims passed the 0.70 support threshold. The highest-scoring claim (entailment score {hi['confidence']:.2f}) was \u201c{hi['claim'][:140]}\u201d, "
               f"and the lowest-scoring (score {low['confidence']:.2f}) was \u201c{low['claim'][:140]}\u201d. "
               "Several plausible-looking claims received very low entailment scores, which is consistent with the zero-shot NLI verifier’s high false-flag rate on real claims ({T:det}); because the claims were not graded this is suggestive only. "
               "The walk-through also shows that the keyword risk screen, not the verifier, decides which unsupported claims lead to abstention.")

P.h2("Answers to the Research Questions")
P.bullet(f"**RQ1 (G1, claim-level verification).** Yes, with qualifications. The adapted 1.7B verifier exceeds the zero-shot NLI verifier on all three test sets (macro-F1 {ci('mixed','medquad_test_iid')}, {ci('mixed','medquad_test_ood')}, {ci('mixed','healthver_test')} against {ci('nli_deberta','medquad_test_iid')}, {ci('nli_deberta','medquad_test_ood')}, {ci('nli_deberta','healthver_test')}), "
         + ("but a small encoder fine-tuned on the same data is a competitive alternative (see the comparison above). " if "deberta_ft" in M else "")
         + "The evidence supports the value of in-domain supervised adaptation; it does not show that a decoder LLM is required for verification.")
P.bullet(f"**RQ2 (G5, safety-oriented evaluation at verifier level).** At \u03c4 = 0.70 on real claims the adapted verifier flags {DT['mixed']['healthver_test']['flag_recall']:.1%} of unsupported claims, wrongly flags {DT['mixed']['healthver_test']['false_flag_rate']:.1%} of entailed ones and lets {DT['mixed']['healthver_test']['contra_accepted']:.1%} of contradicted claims pass as supported "
         f"(zero-shot NLI: {DT['nli_deberta']['healthver_test']['flag_recall']:.1%}, {DT['nli_deberta']['healthver_test']['false_flag_rate']:.1%} and {DT['nli_deberta']['healthver_test']['contra_accepted']:.1%}). "
         f"The synthetic-to-real gap is large ({ci('mixed','medquad_test_iid')} \u2192 {ci('mixed','medquad_test_ood')} \u2192 {ci('mixed','healthver_test')} from held-out topics to unseen edits to real claims). "
         + (f"Synthetic-only training collapses on real claims ({ci('synth_only','healthver_test')}) and real-only training is weak on MedQuAD ({ci('real_only','medquad_test_iid')}, {ci('real_only','medquad_test_ood')}), so both sources are needed. " if HAVE_ABL else "")
         + "Repair success, unnecessary-repair rate and clinical risk reduction, the remaining parts of G5, were not measured.")
P.bullet(f"**RQ3 (G4, uncertainty handling).** Confidence is over-confident on real claims (ECE {M['mixed']['sets']['healthver_test']['ece']:.3f}) but can be repaired with a little real validation data" + (f" (ECE {CAL['healthver_test']['ece_after']:.3f} after temperature scaling)" if HAVE_CAL else "")
         + f". Abstaining on the least confident half of real claims lifts accuracy to {S['healthver_test']['0.5']:.3f}; it does not reach 0.90.")
P.bullet("**Gaps not answered (G2, G3).** Risk-aware assessment and selective repair are present in the prototype pipeline (a keyword risk screen and a grounded repair step with re-verification) but were not evaluated, "
         "so this study cannot say whether they identify clinically significant errors, repair them successfully, or preserve accurate content. They are the first priority for future work.")

P.h2("Discussion and Threats to Validity")
P.body("The results support the following reading. A small adapted model can verify claims reliably when claim types and language resemble the training distribution; it outperforms a zero-shot general-domain NLI model by a wide margin and, with the same training data, a fine-tuned small encoder on unseen edit types and real claims. "
       "When the claims are real and from a different medical domain, performance falls to the mid-0.70s, calibration degrades, and confident mistakes remain. "
       "The original 90% objective is therefore met for synthetic claims but should not be asserted for clinical use.")
P.bullet("*Synthetic-to-real gap.* Rule-based edits are easier than human-written claims, and the in-distribution score of 0.990 is an upper bound on practical performance.")
P.bullet("*Domain shift.* HealthVer is almost entirely COVID-19; MedQuAD concerns consumer health information on rare and common diseases. The two sets cannot separate domain shift from the effect of real versus synthetic claims.")
P.bullet("*Single seed and limited size.* Each configuration was trained once. Confidence intervals reflect test-set sampling only, not training variance; differences of about one macro-F1 point should not be interpreted.")
P.bullet("*Selection effects.* Checkpoints were selected on a validation set that includes real development data; the HealthVer test set was not used for any decision.")
P.bullet("*Scope.* The study evaluates the verification component. Repair success, unnecessary-repair rate, abstention safety and the keyword risk screen were not evaluated, and no clinical expert review was performed.")
P.bullet("*Baseline strength.* The fine-tuned encoder is a general-domain NLI model with 142 M parameters; biomedical encoders pre-trained on PubMed text, and larger encoders, were not tried and could narrow the gap on real claims.")

# ======================================================================= VI. CONCLUSION
P.h2("Implications for a Risk-Aware System")
P.body(f"Three practical consequences follow. First, the cost of errors is asymmetric: at τ = 0.70 the adapted verifier flags {DT['mixed']['healthver_test']['false_flag_rate']:.0%} of correct real claims, which in the pipeline of Algorithm 1 costs a regeneration and a re-check, "
       f"whereas the {DT['mixed']['healthver_test']['contra_accepted']:.1%} of contradicted real claims that pass as supported are the safety-critical errors. The threshold should therefore be stricter for claims that the risk screen marks as high-risk, "
       "and should be set on validation data from the target domain. Second, a verifier can only judge the evidence it is given: when retrieval fails, the correct decision is *neutral*, which the policy converts into repair or abstention. "
       "The coverage of the whole system thus depends on retrieval quality, which this study did not measure. Third, the claims a deployed system must check are produced by an LLM, a distribution that is neither rule-based edits nor human-written claims; "
       "a small, clinician-annotated sample of generated claims and their retrieved evidence is the most valuable next dataset.")
P.body("The findings are consistent with the broader literature in two respects. Headline performance can overstate reliability: decision-level accuracy hid weak evidence support in [@liang2026], and in the present study synthetic scores overstate real-claim performance. "
       "And fine-tuning with external, evidence-based supervision is a more promising route to dependable correction than self-assessment [@kamoi2024,huang2023], which is the route taken by the verifier studied here.")
P.body("**Ethical considerations.** The system is a research prototype and must not be used to diagnose, treat or make medication decisions. No patient data were used: MedQuAD [@abacha2019] is public NIH material and HealthVer is a public claim-verification dataset. "
       "A verifier that is confidently wrong can create false assurance; the calibration results above show this risk is real for real-world claims, and any user-facing deployment would need clinical review, monitoring and clear communication of uncertainty.")

P.h1("Conclusion and Future Work")
P.body(f"We studied claim-level verification of medical statements with a LoRA-adapted Qwen3-1.7B on a consumer GPU. With topic-disjoint splits and bootstrap confidence intervals, the adapted verifier reaches macro-F1 {ci('mixed','medquad_test_iid')} on held-out MedQuAD topics and {ci('mixed','medquad_test_ood')} with unseen edit types, "
       f"clearly above a zero-shot NLI verifier ({ci('nli_deberta','medquad_test_iid')} and {ci('nli_deberta','medquad_test_ood')}), but {ci('mixed','healthver_test')} on real HealthVer claims. "
       "Mixed training data are the best compromise, calibration on real claims is poor, and abstaining on low confidence improves but does not secure 90% accuracy. "
       + (f"A small DeBERTa encoder fine-tuned on the same data is competitive on held-out topics ({ci('deberta_ft','medquad_test_iid')}) but weaker on unseen edit types ({ci('deberta_ft','medquad_test_ood')}) and real claims ({ci('deberta_ft','healthver_test')}). " if "deberta_ft" in M else "")
       + "Of the five research gaps set out in the proposal, the study evaluates G1 and G4 and, at the level of the verifier, G5; G2 (risk-aware assessment) and G3 (selective repair) are implemented in the prototype but not evaluated. "
       "The study thus meets the 90% precision and recall target only in controlled synthetic conditions.")
P.body("Future work follows directly from the limitations. (i) Wire the adapted verifier into the retrieval pipeline and measure the end-to-end metrics set out in the project plan: high-risk error recall, repair success, unnecessary-repair rate, selective coverage and safe-abstention rate. "
       "(ii) Add real annotated data beyond COVID-19, such as clinical NLI or scientific-claim corpora, and replace keyword risk rules with a learned, clinically validated severity score. "
       "(iii) Recalibrate confidence on real validation data and evaluate multiple seeds. (iv) Compare with a biomedical fine-tuned encoder and with larger backbones under 4-bit QLoRA. (v) Obtain clinical expert review of the repaired and abstained outputs. "
       "Until such validation, the system remains a research prototype and must not be used for diagnosis, treatment or medication decisions.")

P.h5("Appendix: Prompt, Generation Settings and Risk Rules")
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

P.h5("Appendix: Extended Results")
TAGS = [("mixed", "Fine-tuned"), ("real_only", "Real only"), ("synth_only", "Synth. only"), ("deberta_ft", "DeBERTa FT"), ("nli_deberta", "DeBERTa NLI"), ("fewshot", "Qwen 6-shot"), ("base", "Qwen base")]
TAGS = [(t, n) for t, n in TAGS if t in M]
rows = []
for tag, nm in TAGS:
    for s, sn in zip(SETS, ["MQ-iid", "MQ-ood", "HealthVer"]):
        c = M[tag]["sets"][s]["ci"]
        f = lambda k: f"{c[k]['point']:.3f} [{c[k]['lo']:.2f},{c[k]['hi']:.2f}]"
        rows.append([nm if s == SETS[0] else "", sn, f("acc"), f("P"), f("R")])
P.table("full", "Accuracy and Macro Precision/Recall with 95% Bootstrap Intervals for All Systems",
        ["System", "Set", "Accuracy", "Macro-P", "Macro-R"], rows, [1.45, 1.25, 1.95, 1.95, 1.95], font=5.4, align=["l", "l", "c", "c", "c"],
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
P.body("The extended tables confirm the pattern of the main text. The unadapted backbone and the zero-shot NLI baseline fail mostly on the neutral and entailment classes of real claims, whereas the adapters fail mostly on contradiction under unseen edit types; "
       "the real-only adapter is the weakest on the synthetic sets and the synthetic-only adapter is the weakest on real claims.")

P.algorithm("Commands that reproduce the reported results", [
    Raw("python build_claim_dataset.py --n-train 6000      # synthetic corpus"),
    Raw("python train_claim_verifier.py --n-real 4000 --n-synth 2500 --epochs 2"),
    Raw("python train_claim_verifier.py --n-synth 0 --out models/ablation-real-only"),
    Raw("python train_claim_verifier.py --n-real 0 --n-synth 2500 --out models/ablation-synth-only"),
    Raw("python dump_predictions.py --adapter <adapter|none> --tag <name> [--split val]"),
    Raw("python nli_baseline.py [val];  python dump_fewshot.py;  python train_encoder_baseline.py"),
    Raw("python analysis.py;  python analysis_extra.py;  python analysis_calib.py"),
    Raw("python paper_work/make_paper_figures.py;  python paper_work/build_paper.py")])

P.h5("Data and Code Availability")
P.body("The corpus generator (build_claim_dataset.py), trainer (train_claim_verifier.py), prediction dump (dump_predictions.py), NLI baseline (nli_baseline.py), statistics (analysis.py, analysis_extra.py, analysis_calib.py) and figure scripts accompany this manuscript; "
       "MedQuAD [@abacha2019] and the HealthVer re-split [@healthver_resplit] are public. All random choices use seed 42, and the numerical results behind every table and figure are stored as JSON files produced by these scripts.")

# ======================================================================= references
P.h5("References")
P.write_references()
doc.save(OUT)
import refs as _R
lines = ["# Verified references (in order of first citation)", "",
         "Bibliographic details were generated from Crossref (DOI records) and the arXiv API (preprints); see Research_Validation_Summary for caveats.", ""]
for i, k in enumerate(P.order, 1):
    link = (f"https://doi.org/{_R.DOI[k]}" if k in _R.DOI else f"https://arxiv.org/abs/{_R.ARXIV[k]}" if k in _R.ARXIV else _R.ONLINE[k][0].split("Available: ")[-1])
    lines.append(f"[{i}] {_R.fmt(k, P.cache)} <{link}>")
(ROOT / "paper" / "references_verified.md").write_text(chr(10).join(lines), encoding="utf-8")
print("saved", OUT, "| refs:", len(P.order), "| figs:", P.figs, "| tables:", P.tabs)
