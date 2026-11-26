"""
Generate expanded fine-tuning dataset from MedQuAD and PubMedQA.

Run:  python generate_finetune_data.py
Out:  data/finetune_examples.json  (~100+ examples across 3 tasks)

Tasks:
  1. claim_verification  — entailment / neutral / contradiction
  2. grounded_repair     — fix an inaccurate draft using evidence
  3. abstention          — high-risk claim with no evidence → refuse & redirect
"""
from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

random.seed(42)

DATA      = Path(__file__).resolve().parent.parent / "data"
MEDQUAD   = DATA / "MedQuAD"
PUBMEDQA  = DATA / "PubMedQA" / "data" / "ori_pqal.json"
OUT       = DATA / "finetune_examples.json"

# ---------------------------------------------------------------------------
# Shared instructions (must match what pipeline uses at inference time)
# ---------------------------------------------------------------------------
INSTR_VERIFY  = (
    "Classify whether the evidence supports, is neutral to, or contradicts the claim. "
    "Reply with one word: entailment, neutral, or contradiction."
)
INSTR_REPAIR  = (
    "Revise the draft answer to be accurate and grounded in the evidence. "
    "Keep the response concise and cite the evidence."
)
INSTR_ABSTAIN = (
    "Decide whether to answer or abstain. If the claim involves high-risk medication or "
    "diagnosis advice and there is no supporting evidence, output the abstention message."
)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _extract(node) -> str:
    return " ".join(t.strip() for t in node.itertext() if t.strip())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if len(s.split()) >= 6]


def load_medquad(max_pairs: int = 1200) -> list[dict]:
    pairs: list[dict] = []
    for p in sorted(MEDQUAD.rglob("*.xml")):
        try:
            root = ET.parse(p).getroot()
            for pair in root.findall(".//QAPair") or [root]:
                q = " ".join(_extract(x) for x in pair.findall(".//Question"))
                a = " ".join(_extract(x) for x in pair.findall(".//Answer"))
                if q and len(a) >= 100:
                    pairs.append({"q": q.strip(), "a": a.strip(), "source": p.name})
        except Exception:
            continue
        if len(pairs) >= max_pairs:
            break
    return pairs


def load_pubmedqa(max_samples: int = 300) -> list[dict]:
    if not PUBMEDQA.exists():
        return []
    data = json.loads(PUBMEDQA.read_text(encoding="utf-8"))
    out: list[dict] = []
    for uid, item in data.items():
        if not isinstance(item, dict):
            continue
        q   = item.get("QUESTION", "")
        ctx = item.get("CONTEXTS", [])
        lbl = str(item.get("final_decision", "")).strip().lower()
        if q and ctx and lbl in {"yes", "no", "maybe"}:
            out.append({"q": q, "context": " ".join(str(c) for c in ctx)[:600], "label": lbl})
        if len(out) >= max_samples:
            break
    return out

# ---------------------------------------------------------------------------
# Task 1 — Claim Verification
# ---------------------------------------------------------------------------

def make_entailment(pairs: list[dict], n: int = 20) -> list[dict]:
    """Claim from same QA pair → entailment."""
    out = []
    for p in random.sample(pairs, min(n * 4, len(pairs))):
        sents = _sentences(p["a"])
        if not sents:
            continue
        out.append({
            "task": "claim_verification",
            "instruction": INSTR_VERIFY,
            "input": f"Claim: {sents[0]}\nEvidence: {p['a'][:500]}",
            "output": "entailment",
        })
        if len(out) >= n:
            break
    return out


def make_neutral(pairs: list[dict], n: int = 15) -> list[dict]:
    """Claim from one topic, evidence from a different topic → neutral."""
    out = []
    shuffled = pairs[:]
    random.shuffle(shuffled)
    half = len(shuffled) // 2
    for i in range(min(n * 2, half)):
        p1, p2 = shuffled[i], shuffled[half + i]
        if p1["source"] == p2["source"]:
            continue
        sents = _sentences(p1["a"])
        if not sents:
            continue
        out.append({
            "task": "claim_verification",
            "instruction": INSTR_VERIFY,
            "input": f"Claim: {sents[0]}\nEvidence: {p2['a'][:400]}",
            "output": "neutral",
        })
        if len(out) >= n:
            break
    return out


def make_pubmedqa_verification(samples: list[dict], n: int = 20) -> list[dict]:
    """PubMedQA yes/no/maybe → entailment/contradiction/neutral."""
    lmap = {"yes": "entailment", "no": "contradiction", "maybe": "neutral"}
    out = []
    for s in random.sample(samples, min(n, len(samples))):
        out.append({
            "task": "claim_verification",
            "instruction": INSTR_VERIFY,
            "input": f"Claim: {s['q']}\nEvidence: {s['context']}",
            "output": lmap[s["label"]],
        })
    return out

# ---------------------------------------------------------------------------
# Task 2 — Grounded Repair (high-quality hand-crafted + MedQuAD-based)
# ---------------------------------------------------------------------------

HARDCODED_REPAIRS = [
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: What is the first-line treatment for type 2 diabetes?\n"
            "Draft: The first-line treatment for type 2 diabetes is always insulin injection.\n"
            "Evidence: Metformin is recommended as the first-line pharmacological therapy for type 2 diabetes "
            "by most major clinical guidelines. Lifestyle modifications including diet and exercise are also "
            "foundational. Insulin is typically reserved for patients with very high blood glucose or who fail "
            "oral agents."
        ),
        "output": (
            "The first-line pharmacological treatment for type 2 diabetes is metformin, alongside lifestyle "
            "modifications such as diet and exercise. Insulin is not the initial therapy for most patients; "
            "it is typically reserved for those with very high blood glucose levels or who do not respond "
            "adequately to oral medications."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: Can antibiotics treat the flu?\n"
            "Draft: Yes, antibiotics are effective and should be taken at the first sign of flu.\n"
            "Evidence: Influenza is caused by a virus. Antibiotics treat bacterial infections only and have "
            "no effect on viral illnesses such as influenza. Inappropriate antibiotic use contributes to "
            "antibiotic resistance. Antiviral medications such as oseltamivir may reduce flu duration "
            "if started early."
        ),
        "output": (
            "Antibiotics are not effective against the flu because influenza is a viral infection, not bacterial. "
            "Using antibiotics for viral illnesses is inappropriate and contributes to antibiotic resistance. "
            "Antiviral medications such as oseltamivir may shorten flu duration when started within the first "
            "48 hours of symptom onset."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: What causes most stomach ulcers?\n"
            "Draft: Stress and spicy food are the primary causes of stomach ulcers.\n"
            "Evidence: Most peptic ulcers are caused by Helicobacter pylori bacterial infection or long-term "
            "NSAID use (e.g., aspirin, ibuprofen). Stress and spicy food do not directly cause ulcers, though "
            "they may worsen symptoms. Treatment targets H. pylori eradication or NSAID cessation."
        ),
        "output": (
            "Most stomach ulcers are caused by Helicobacter pylori bacterial infection or long-term use of "
            "NSAIDs such as aspirin or ibuprofen. Stress and spicy food do not directly cause ulcers, "
            "though they may aggravate symptoms. Treatment focuses on eradicating H. pylori with antibiotics "
            "or discontinuing NSAIDs."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: Is it safe to take ibuprofen during pregnancy?\n"
            "Draft: Ibuprofen is a safe and effective pain reliever throughout pregnancy.\n"
            "Evidence: NSAIDs including ibuprofen are generally contraindicated in pregnancy, especially in "
            "the third trimester, due to risks of premature closure of the ductus arteriosus, fetal renal "
            "impairment, and reduced amniotic fluid. Paracetamol (acetaminophen) is preferred for pain relief "
            "during pregnancy."
        ),
        "output": (
            "Ibuprofen is not safe during pregnancy. NSAIDs including ibuprofen are contraindicated, especially "
            "in the third trimester, due to risks of premature closure of the ductus arteriosus, fetal kidney "
            "problems, and reduced amniotic fluid. Paracetamol (acetaminophen) is the recommended alternative "
            "for pain relief during pregnancy."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: What is the best way to treat a broken bone at home?\n"
            "Draft: Most broken bones can be treated at home with ice and rest without needing a doctor.\n"
            "Evidence: Fractures require medical evaluation to confirm diagnosis via X-ray and to determine "
            "appropriate treatment, which may include casting, splinting, or surgery for complex breaks. "
            "Delayed treatment can lead to improper healing. Home management without diagnosis is inappropriate."
        ),
        "output": (
            "Broken bones require prompt medical evaluation, including X-ray imaging, to confirm the fracture "
            "and determine treatment. Treatment may range from casting or splinting to surgery for complex "
            "fractures. Home management without proper diagnosis is inappropriate and can lead to improper "
            "healing or complications."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: Does high blood pressure always cause symptoms?\n"
            "Draft: High blood pressure always causes headaches, dizziness, and nosebleeds, making it easy to detect.\n"
            "Evidence: Hypertension is known as the 'silent killer' because it usually produces no symptoms "
            "until serious complications such as heart attack, stroke, or kidney disease occur. Regular blood "
            "pressure monitoring is essential for early detection."
        ),
        "output": (
            "High blood pressure usually causes no symptoms, which is why it is called the 'silent killer.' "
            "Symptoms such as headaches and dizziness are not reliable indicators and may appear only when "
            "complications arise. Regular blood pressure checks are essential for early detection and "
            "prevention of complications like heart attack and stroke."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: Is depression just a matter of willpower?\n"
            "Draft: Depression is a character weakness and can be overcome simply by thinking positively.\n"
            "Evidence: Major depressive disorder is a recognized medical condition with biological, "
            "psychological, and social contributors, including altered brain chemistry, genetics, and life "
            "events. It often requires professional treatment including psychotherapy, medication, or both. "
            "Willpower alone is not sufficient treatment."
        ),
        "output": (
            "Depression is not a matter of willpower or character weakness. It is a recognized medical "
            "condition with complex biological, psychological, and social factors, including alterations in "
            "brain chemistry and genetics. Effective treatment typically includes psychotherapy, medication, "
            "or a combination, and professional support is important for recovery."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: Can vitamin C supplements prevent the common cold?\n"
            "Draft: Taking vitamin C supplements will prevent you from getting a cold.\n"
            "Evidence: Regular vitamin C supplementation does not prevent colds in the general population. "
            "Evidence suggests it may modestly reduce cold duration (by about 8–14%) and severity in some "
            "people. Very high doses may cause gastrointestinal side effects."
        ),
        "output": (
            "Vitamin C supplementation does not prevent colds in the general population. However, evidence "
            "suggests it may modestly reduce the duration and severity of colds in some individuals. "
            "Very high doses of vitamin C can cause gastrointestinal side effects and should be avoided."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: Is a low-carb diet good for people with kidney disease?\n"
            "Draft: A high-protein, low-carbohydrate diet is ideal for all people with kidney disease.\n"
            "Evidence: For most patients with chronic kidney disease, a low-protein diet is recommended to "
            "reduce the accumulation of nitrogen waste products that the kidneys must filter. High-protein "
            "diets can accelerate kidney damage. Dietary management should be individualized by a renal dietitian."
        ),
        "output": (
            "High-protein diets are generally not appropriate for people with chronic kidney disease. A "
            "low-protein diet is typically recommended to reduce the burden of nitrogen waste on the kidneys. "
            "High-protein intake can accelerate kidney damage. Dietary plans should be individualized and "
            "managed with the guidance of a renal dietitian."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: Does exercise help with anxiety?\n"
            "Draft: Exercise has no effect on anxiety and people with anxiety should rest instead.\n"
            "Evidence: Regular physical exercise has been shown to reduce symptoms of anxiety and depression. "
            "Aerobic exercise in particular can reduce stress hormones, improve mood, and serve as an "
            "effective complementary treatment for anxiety disorders alongside therapy and medication."
        ),
        "output": (
            "Regular physical exercise has well-documented benefits for anxiety. Aerobic exercise in "
            "particular can reduce stress hormones, improve mood, and serve as an effective complementary "
            "treatment alongside therapy and medication. Rest alone is not the recommended approach for "
            "managing anxiety."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: What are the warning signs of a stroke?\n"
            "Draft: A stroke usually causes a gradual headache over several days and is not a medical emergency.\n"
            "Evidence: Stroke warning signs appear suddenly and include sudden numbness or weakness in the face, "
            "arm, or leg (especially on one side), sudden confusion or trouble speaking, sudden vision problems, "
            "sudden severe headache with no known cause, and sudden trouble walking. The FAST acronym "
            "(Face drooping, Arm weakness, Speech difficulty, Time to call 911) is widely recommended."
        ),
        "output": (
            "Stroke symptoms appear suddenly and are a medical emergency. Warning signs include sudden numbness "
            "or weakness (especially on one side of the body), sudden confusion or difficulty speaking, sudden "
            "vision problems, sudden severe headache, and sudden trouble walking. Use the FAST acronym — Face "
            "drooping, Arm weakness, Speech difficulty, Time to call 911 — and seek emergency care immediately."
        ),
    },
    {
        "task": "grounded_repair", "instruction": INSTR_REPAIR,
        "input": (
            "Question: Is it safe to stop blood pressure medication once blood pressure normalizes?\n"
            "Draft: Once your blood pressure is normal, you can safely stop taking your medication.\n"
            "Evidence: Antihypertensive medications control but do not cure high blood pressure. Blood pressure "
            "typically returns to elevated levels when medication is stopped. Stopping medication without "
            "medical guidance can cause rebound hypertension and increase risk of heart attack and stroke."
        ),
        "output": (
            "Blood pressure medications control hypertension but do not cure it. If medication is stopped, "
            "blood pressure typically rises again. Stopping antihypertensive medication without medical "
            "guidance can cause rebound hypertension and significantly increase the risk of heart attack "
            "and stroke. Any changes to medication should be discussed with your doctor."
        ),
    },
]


def make_grounded_repair_from_medquad(pairs: list[dict], n: int = 10) -> list[dict]:
    """Generate additional repair examples from MedQuAD content."""
    out = []
    wrong_openers = [
        "Studies have conclusively shown that",
        "It is a well-known fact that",
        "Medical consensus is that",
        "Research has proven beyond doubt that",
    ]
    wrong_closers = [
        " This always resolves completely on its own without any medical intervention.",
        " This is completely safe for all patients regardless of other health conditions.",
        " No further treatment or monitoring is needed.",
    ]
    for p in random.sample(pairs, min(n * 5, len(pairs))):
        sents = _sentences(p["a"])
        if len(sents) < 2:
            continue
        opener  = random.choice(wrong_openers)
        closer  = random.choice(wrong_closers)
        draft   = f"{opener} {sents[0].lower()}{closer}"
        correct = " ".join(sents[:2])
        evidence = p["a"][:500]
        out.append({
            "task": "grounded_repair",
            "instruction": INSTR_REPAIR,
            "input": f"Question: {p['q']}\nDraft: {draft}\nEvidence: {evidence}",
            "output": correct,
        })
        if len(out) >= n:
            break
    return out

# ---------------------------------------------------------------------------
# Task 3 — Abstention
# ---------------------------------------------------------------------------

HIGH_RISK = [
    ("You should stop taking your blood pressure medication if you feel dizzy.",
     "I can't verify this potentially high-risk medication claim. Stopping blood pressure medication suddenly can cause dangerous rebound hypertension. Please consult your prescribing physician before changing your medication."),
    ("Taking a double dose of insulin will help control high blood sugar faster.",
     "I can't verify this high-risk medication claim. Doubling insulin doses without medical guidance can cause life-threatening hypoglycaemia. Please contact your diabetes care team immediately if your blood sugar is uncontrolled."),
    ("It is safe to combine any over-the-counter painkillers for stronger pain relief.",
     "I can't verify this claim. Combining painkillers can cause serious harm including gastrointestinal bleeding and liver damage. Please consult a pharmacist or doctor before combining medications."),
    ("Stopping chemotherapy early once you feel better will not affect your cancer treatment.",
     "I can't verify this high-risk medical claim. Stopping chemotherapy prematurely may allow cancer cells to regrow and reduce treatment effectiveness. Please discuss any concerns about your treatment with your oncologist."),
    ("You should take aspirin every day to prevent a heart attack even if your doctor has not recommended it.",
     "I can't verify this claim for your specific situation. Daily aspirin can cause serious bleeding risks and is only appropriate for certain patients. Please consult your doctor before starting daily aspirin therapy."),
    ("You can safely share your prescription medication with someone who has the same symptoms.",
     "I can't verify this claim. Sharing prescription medications is dangerous and illegal. Medications are prescribed based on individual health factors; what is safe for one person may be harmful for another. Please consult a healthcare professional."),
    ("You can accurately diagnose a heart attack based on symptoms alone and treat it at home.",
     "I can't verify this potentially life-threatening claim. Suspected heart attack symptoms require immediate emergency care — call 911. Early medical intervention is critical for minimizing heart muscle damage."),
    ("Stopping seizure medication on your own when seizures stop is safe.",
     "I can't verify this high-risk claim. Stopping antiepileptic drugs without medical guidance can trigger breakthrough seizures that may be life-threatening. Please consult your neurologist before changing your seizure medications."),
    ("Children can safely take adult dosages of over-the-counter medications.",
     "I can't verify this potentially dangerous claim. Adult dosages can be harmful or fatal for children due to differences in body weight and metabolism. Always use pediatric formulations and dosages, and consult a paediatrician if unsure."),
    ("You can treat a severe allergic reaction at home with antihistamines alone.",
     "I can't verify this high-risk claim. Severe allergic reactions (anaphylaxis) require immediate epinephrine injection and emergency medical care. Antihistamines are not sufficient for anaphylaxis. Call emergency services immediately."),
    ("Stopping antidepressants suddenly is safe once you feel better.",
     "I can't verify this high-risk claim. Abruptly stopping antidepressants can cause discontinuation syndrome and may lead to relapse of depression. Please work with your prescribing doctor to taper medication safely."),
    ("Herbal supplements are always safe to take alongside prescription medications.",
     "I can't verify this claim. Many herbal supplements interact significantly with prescription medications. For example, St. John's Wort reduces the effectiveness of several drugs. Please consult your pharmacist before combining supplements with medications."),
    ("It is safe to drink alcohol while taking any type of antibiotics.",
     "I can't verify this claim. Alcohol interacts with certain antibiotics causing serious reactions such as severe nausea and vomiting, particularly with metronidazole and tinidazole. Please consult your pharmacist about alcohol interactions with your specific antibiotic."),
    ("Using someone else's insulin as a temporary measure is acceptable.",
     "I can't verify this high-risk claim. Using another person's insulin without medical supervision is dangerous; insulin types and doses are highly individual. Please contact your doctor or go to an emergency department if you cannot access your prescribed insulin."),
    ("You can safely diagnose your own skin condition and treat it at home.",
     "I can't verify this claim. Skin conditions can range from minor to serious, including melanoma. Self-diagnosis may delay treatment of significant conditions. Please consult a dermatologist for proper evaluation."),
    ("All pain is best treated with the strongest available opioid medication.",
     "I can't verify this claim. Opioids carry significant risks including addiction and overdose and are not appropriate for all types of pain. Pain management should be individualized and supervised by a healthcare provider."),
    ("You can use an old prescription from a previous illness without consulting a doctor.",
     "I can't verify this claim for your current situation. Using old prescriptions is risky; your condition may differ, the medication may have expired, or interactions with other medications may exist. Please consult a healthcare provider."),
    ("High doses of vitamin D are safe and beneficial for everyone.",
     "I can't verify this claim. Excessive vitamin D can cause hypercalcaemia, with symptoms including nausea, weakness, and kidney problems. Supplementation should be guided by measured blood levels and a healthcare provider's recommendation."),
    ("You should take corticosteroid cream more liberally than prescribed to speed healing.",
     "I can't verify this high-risk medication claim. Overusing topical corticosteroids can cause skin thinning, adrenal suppression, and other serious effects. Please follow your prescribed dosage and consult your doctor if it is not providing sufficient relief."),
    ("Stopping a course of antibiotics early once symptoms resolve is harmless.",
     "I can't verify this claim. Stopping antibiotics before completing the full course can lead to treatment failure and contribute to antibiotic resistance. Please follow your prescribed regimen and consult your doctor before stopping."),
    ("It is safe to self-administer intravenous fluids at home without medical supervision.",
     "I can't verify this high-risk claim. Intravenous administration requires sterile technique and medical supervision; errors can cause infection, air embolism, or electrolyte imbalances. Please seek care through appropriate medical channels."),
    ("You can delay seeking care for chest pain to see if it resolves on its own.",
     "I can't verify this potentially life-threatening claim. Chest pain can indicate a heart attack or other serious condition requiring immediate medical attention. Please call 911 or go to an emergency department without delay."),
]


def make_abstention(n: int = 22) -> list[dict]:
    sampled = random.sample(HIGH_RISK, min(n, len(HIGH_RISK)))
    return [
        {
            "task": "abstention",
            "instruction": INSTR_ABSTAIN,
            "input": f"Claim: {claim}\nEvidence: [none]",
            "output": resp,
        }
        for claim, resp in sampled
    ]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[generate] Loading MedQuAD ...")
    pairs = load_medquad(max_pairs=1200)
    print(f"[generate] {len(pairs)} MedQuAD Q&A pairs loaded.")

    print("[generate] Loading PubMedQA ...")
    pubmed = load_pubmedqa(max_samples=300)
    print(f"[generate] {len(pubmed)} PubMedQA samples loaded.")

    examples: list[dict] = []

    print("[generate] Building claim_verification examples ...")
    examples += make_entailment(pairs,            n=22)
    examples += make_neutral(pairs,               n=15)
    examples += make_pubmedqa_verification(pubmed, n=20) if pubmed else []

    print("[generate] Building grounded_repair examples ...")
    examples += HARDCODED_REPAIRS
    examples += make_grounded_repair_from_medquad(pairs, n=10)

    print("[generate] Building abstention examples ...")
    examples += make_abstention(n=22)

    random.shuffle(examples)

    OUT.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = Counter(e["task"] for e in examples)
    print(f"\n[generate] Saved {len(examples)} examples to {OUT}")
    for task, cnt in counts.items():
        print(f"  {task}: {cnt}")


if __name__ == "__main__":
    main()
