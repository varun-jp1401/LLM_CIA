"""
Build a balanced 3-way claim-verification dataset (entailment / neutral / contradiction)
from MedQuAD, matching the project's "claim verification F1" objective.

Anti-memorisation design
------------------------
* Splits are by DOCUMENT (MedQuAD file / <Focus>), so held-out topics never appear in training.
* Label-preserving edits (truncation, synonym swaps, prefixes) are applied at the same rate to
  ALL classes, so "claim is a verbatim substring of evidence" is not a shortcut for entailment.
* Contradictions use several edit families; neutral uses hard negatives (same document, other
  section), easy negatives (other document) and "unsupported addition" claims.
* `test_ood` uses edit families / addition templates that never occur in train/val.

Real data (HealthVer) is converted separately by the trainer; see train_claim_verifier.py.

Usage:  python build_claim_dataset.py --n-train 6000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
MEDQUAD = ROOT / "data" / "MedQuAD"
OUT = ROOT / "data" / "claim_verification"

LABELS = ("entailment", "neutral", "contradiction")

# ---------------------------------------------------------------- text utils
def _text(node) -> str:
    return " ".join(t.strip() for t in node.itertext() if t.strip())


def _clean(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"\bKey Points\b", " ", answer)
    answer = re.sub(r"\s-\s", ". ", answer)  # bullet markers
    answer = re.sub(r"\(\s*\)", "", answer)
    return answer.strip()


def sentences(text: str) -> list[str]:
    out = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        s = s.strip(" -*")
        n = len(s.split())
        if 7 <= n <= 45 and s[-1:] in ".!?" and "http" not in s and not re.search(r"[A-Z]{3,}\s+[A-Z]{3,}\s+[A-Z]{3,}", s):
            out.append(s)
    return out


def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a: str, b: str) -> float:
    A, B = _tok(a), _tok(b)
    return len(A & B) / max(1, len(A | B))


# ------------------------------------------------------------ edit families
def _swap_map(text: str, table: dict[str, str]) -> str | None:
    """Replace the first word that appears in `table` (whole-word, case aware)."""
    words = list(table)
    random.shuffle(words)
    for w in words:
        m = re.search(rf"\b{re.escape(w)}\b", text, flags=re.I)
        if m:
            rep = table[w]
            if m.group(0)[0].isupper():
                rep = rep[0].upper() + rep[1:]
            return text[:m.start()] + rep + text[m.end():]
    return None


def _both(pairs: list[tuple[str, str]]) -> dict[str, str]:
    d = {}
    for a, b in pairs:
        d[a] = b
        d[b] = a
    return d


ANTONYMS = _both([
    ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
    ("higher", "lower"), ("more", "less"), ("before", "after"), ("safe", "unsafe"),
    ("benign", "malignant"), ("common", "rare"), ("commonly", "rarely"),
    ("improves", "worsens"), ("reduces", "increases"), ("reduce", "increase"),
    ("effective", "ineffective"), ("can", "cannot"), ("with", "without"),
    ("causes", "prevents"), ("cause", "prevent"), ("fast", "slow"), ("rapidly", "slowly"),
    ("large", "small"), ("severe", "mild"), ("early", "late"), ("often", "rarely"),
    ("spread", "stay localized"), ("lose", "gain"), ("loss", "gain"), ("stronger", "weaker"),
    ("healthy", "damaged"), ("positive", "negative"), ("overactive", "underactive"),
])
QUANT = _both([("most", "few"), ("all", "none"), ("many", "few"), ("always", "never"),
               ("everyone", "no one"), ("some", "no"), ("usually", "rarely")])
# held-out (OOD) contradiction family: demographic / directional / temporal swaps
OOD_SWAP = _both([("men", "women"), ("children", "adults"), ("male", "female"), ("left", "right"),
                  ("upper", "lower"), ("acute", "chronic"), ("older", "younger"),
                  ("inherited", "acquired"), ("first", "last"), ("morning", "evening"),
                  ("infants", "teenagers"), ("daily", "monthly")])

MODALS = [("is", "is not"), ("are", "are not"), ("can", "cannot"), ("may", "may not"),
          ("should", "should not"), ("does", "does not"), ("do", "do not"),
          ("has", "has not"), ("have", "have not"), ("will", "will not"), ("was", "was not"),
          ("were", "were not"), ("must", "must not")]


def negate(text: str) -> str | None:
    # avoid double negatives / sentences that already carry negation (ambiguous labels)
    if re.search(r"\b(no|none|neither|nor|nothing|nobody|never|without|not|cannot)\b|n't", text, flags=re.I):
        return None
    if re.search(r"\b(is|are|can|may|should|does|do|has|have|will|was|were|must)\s+(not|never)\b|n't|\bcannot\b", text):
        t = re.sub(r"\b(is|are|can|may|should|does|do|has|have|will|was|were|must)\s+(not|never)\b",
                   lambda m: m.group(1), text, count=1)
        if t != text:
            return t
        return text.replace("cannot", "can", 1) if "cannot" in text else None
    random.shuffle(MODALS)
    for base, neg in MODALS:
        m = re.search(rf"\b{base}\b", text)
        if m and m.start() > 0:
            return text[:m.start()] + neg + text[m.end():]
    return None


def change_number(text: str) -> str | None:
    nums = list(re.finditer(r"\b\d+(?:\.\d+)?\b", text))
    if not nums:
        return None
    m = random.choice(nums)
    v = float(m.group(0))
    if v == 0:
        new = "1"
    else:
        f = random.choice([2, 3, 5, 10, 0.5, 0.25, 0.1])
        nv = v * f
        new = str(int(nv)) if float(int(nv)) == nv else f"{nv:.1f}"
    if new == m.group(0):
        return None
    return text[:m.start()] + new + text[m.end():]


TRAIN_CONTRA = {
    "negation": negate,
    "antonym": lambda t: _swap_map(t, ANTONYMS),
    "number": change_number,
    "quantifier": lambda t: _swap_map(t, QUANT),
}
OOD_CONTRA = {"swap_ood": lambda t: _swap_map(t, OOD_SWAP)}

# ---------------------------------------------- label-preserving surface edits
SAFE_SYN = _both([("also called", "also known as"), ("begin", "start"), ("find", "detect"),
                  ("many", "numerous"), ("get", "obtain"), ("show", "indicate"), ("help", "assist"),
                  ("need", "require"), ("about", "around"), ("part of", "component of"),
                  ("people", "individuals"), ("doctor", "physician"), ("use", "utilize")])
PREFIXES = ["According to the passage, ", "It is stated that ", "The text says that ", "Reportedly, ",
            "As described, ", "The reference notes that ", "In general, "]


def preserving_edit(text: str) -> str:
    """Edits that do not change meaning; applied to claims of ALL classes."""
    r = random.random()
    if r < 0.35:
        t = _swap_map(text, SAFE_SYN)
        if t:
            return t
    if r < 0.60:
        parts = re.split(r"[,;] ", text)
        if len(parts) > 1:
            longest = max(parts, key=lambda p: len(p.split()))
            if len(longest.split()) >= 6:
                longest = longest.strip().rstrip(".")
                return longest[0].upper() + longest[1:] + "."
    return text


def add_prefix(text: str) -> str:
    if random.random() < 0.35:
        p = random.choice(PREFIXES)
        return p + text[0].lower() + text[1:] if not re.match(r"[A-Z]{2,}", text) else p + text
    return text


TRAIN_ADD = [
    "and this usually resolves without any treatment", "and it affects roughly one in three patients",
    "and symptoms typically appear within a few days", "and it is more common in people over 60",
    "and the effect is permanent in most cases", "and it is usually fatal if left untreated",
    "and the standard dose is 500 mg twice daily", "and it was first described in the 1950s",
    "and this has been confirmed in large randomized trials", "and surgery is always required",
]
OOD_ADD = [
    "and this risk is doubled in smokers", "and it responds best to an herbal supplement",
    "and it was linked to a specific gene in 2010", "and vaccination eliminates this entirely",
    "and recovery takes exactly six weeks", "and this is highest during the summer months",
]


def add_unsupported(text: str, bank: list[str]) -> str:
    base = text.rstrip(". ")
    return f"{base} {random.choice(bank)}."


# --------------------------------------------------------------- load corpus
def load_docs() -> list[dict]:
    docs = []
    for path in sorted(MEDQUAD.rglob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        focus = (root.findtext("Focus") or path.stem).strip()
        qa = []
        for pair in root.findall(".//QAPair"):
            q = _text(pair.find("Question")) if pair.find("Question") is not None else ""
            a = _clean(_text(pair.find("Answer"))) if pair.find("Answer") is not None else ""
            sents = sentences(a)
            if len(sents) >= 3:
                qa.append({"question": q, "sents": sents})
        if qa:
            docs.append({"id": f"{path.parent.name}/{path.stem}", "source": path.parent.name,
                         "focus": focus, "qa": qa})
    return docs


def split_docs(docs: list[dict], seed: int):
    """Split by focus so no topic appears in more than one split."""
    by_focus = defaultdict(list)
    for d in docs:
        by_focus[d["focus"].casefold()].append(d)
    keys = sorted(by_focus)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    cut1, cut2 = int(n * .80), int(n * .90)
    parts = {"train": keys[:cut1], "val": keys[cut1:cut2], "test": keys[cut2:]}
    return {k: [d for key in v for d in by_focus[key]] for k, v in parts.items()}


# ---------------------------------------------------------------- generation
def evidence_window(sents: list[str], idx: int, k: int) -> str:
    start = max(0, min(idx - random.randint(0, k - 1), len(sents) - k))
    return " ".join(sents[start:start + k])


def make_example(label: str, docs: list[dict], allowed_contra: dict, add_bank: list[str]) -> dict | None:
    doc = random.choice(docs)
    qa = random.choice(doc["qa"])
    sents = qa["sents"]
    idx = random.randrange(len(sents))
    k = random.choice([2, 3, 3, 4])
    evidence = evidence_window(sents, idx, k)
    target = sents[idx]
    if target not in evidence:
        return None
    meta = {"doc": doc["id"], "focus": doc["focus"], "source": doc["source"]}

    if label == "entailment":
        claim = preserving_edit(target)
        kind = "entail"
    elif label == "contradiction":
        base = preserving_edit(target)
        names = list(allowed_contra)
        random.shuffle(names)
        claim = None
        for name in names:
            claim = allowed_contra[name](base)
            if claim and claim != base:
                kind = name
                break
            claim = None
        if claim is None:
            return None
    else:
        r = random.random()
        if r < 0.40:  # hard: same document, different QA pair
            others = [q for q in doc["qa"] if q is not qa]
            if not others:
                return None
            claim = random.choice(random.choice(others)["sents"])
            kind = "neutral_same_doc"
        elif r < 0.65:  # easy: other document
            other = random.choice(docs)
            if other["id"] == doc["id"]:
                return None
            claim = random.choice(random.choice(other["qa"])["sents"])
            kind = "neutral_other_doc"
        else:  # unsupported addition to a supported claim
            claim = add_unsupported(preserving_edit(target), add_bank)
            kind = "neutral_overclaim"
        if kind != "neutral_overclaim" and (jaccard(claim, evidence) > 0.45 or claim in evidence):
            return None
    claim = add_prefix(claim)
    if label != "neutral" and False:
        return None
    return {"claim": claim, "evidence": evidence, "label": label, "kind": kind, **meta}


def build(docs, n, allowed_contra, add_bank, seed) -> list[dict]:
    random.seed(seed)
    out, seen = [], set()
    per = n // 3
    for label in LABELS:
        got, tries = 0, 0
        while got < per and tries < per * 60:
            tries += 1
            ex = make_example(label, docs, allowed_contra, add_bank)
            if not ex:
                continue
            key = (ex["claim"], ex["evidence"])
            if key in seen:
                continue
            seen.add(key)
            out.append(ex)
            got += 1
    random.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    docs = load_docs()
    parts = split_docs(docs, a.seed)
    print({k: len(v) for k, v in parts.items()}, "documents (split by topic)")
    OUT.mkdir(parents=True, exist_ok=True)

    sets = {
        "train": build(parts["train"], a.n_train, TRAIN_CONTRA, TRAIN_ADD, a.seed),
        "val": build(parts["val"], a.n_val, TRAIN_CONTRA, TRAIN_ADD, a.seed + 1),
        "test_iid": build(parts["test"], a.n_test, TRAIN_CONTRA, TRAIN_ADD, a.seed + 2),
        "test_ood": build(parts["test"], a.n_test, OOD_CONTRA, OOD_ADD, a.seed + 3),
    }
    # the OOD set must not reuse train topics; verify
    train_focus = {d["focus"].casefold() for d in parts["train"]}
    for name in ("val", "test_iid", "test_ood"):
        assert not ({e["focus"].casefold() for e in sets[name]} & train_focus), f"topic leakage in {name}"
    manifest = {}
    for name, rows in sets.items():
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        manifest[name] = {"n": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
                          "kinds": dict(Counter(r["kind"] for r in rows))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    for r in sets["train"][:6]:
        print("\n", r["label"], r["kind"], "\n CLAIM:", r["claim"], "\n EVID :", r["evidence"][:200])


if __name__ == "__main__":
    main()
