"""Shared helpers for claim-verification training / evaluation (Qwen3-1.7B + LoRA)."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
BASE_MODEL = "Qwen/Qwen3-1.7B"
SYN_DIR = ROOT / "data" / "claim_verification"
HV_DIR = ROOT / "data" / "external" / "healthver"
LABELS = ("entailment", "neutral", "contradiction")
HV_MAP = {"Supports": "entailment", "Neutral": "neutral", "Refutes": "contradiction"}

# Same wording as the pipeline's verification instruction (generate_finetune_data.py).
INSTRUCTION = ("Classify whether the evidence supports, is neutral to, or contradicts the claim. "
               "Reply with one word: entailment, neutral, or contradiction.")


def load_synthetic(name: str) -> list[dict]:
    rows = json.loads((SYN_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [{"claim": r["claim"], "evidence": r["evidence"], "label": r["label"], "src": "medquad:" + r["kind"]}
            for r in rows]


def load_healthver(split: str) -> list[dict]:
    path = HV_DIR / f"{split}.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["Label"] in HV_MAP:
            rows.append({"claim": r["Claim"].strip(), "evidence": r["Evidence"].strip(),
                         "label": HV_MAP[r["Label"]], "src": "healthver"})
    return rows


def load_eval_sets(split: str = "test") -> dict[str, list[dict]]:
    """Named evaluation sets: held-out test sets (default) or the validation sets used for model selection."""
    if split == "val":
        return {"medquad_val": load_synthetic("val"), "healthver_dev": load_healthver("dev")}
    return {"medquad_test_iid": load_synthetic("test_iid"), "medquad_test_ood": load_synthetic("test_ood"),
            "healthver_test": load_healthver("test")}


def predictions_path(tag: str, split: str = "test") -> Path:
    return RESULTS_DIR / (f"predictions_{tag}.json" if split == "test" else f"predictions_{tag}_val.json")


def build_user_text(row: dict) -> str:
    return f"{INSTRUCTION}\n\nClaim: {row['claim']}\nEvidence: {row['evidence'][:1800]}"


def encode(tokenizer, rows: list[dict], max_len: int = 448) -> list[list[int]]:
    ids = []
    for r in rows:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_user_text(r)}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        t = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if len(t) > max_len:  # keep the end (generation prompt); trim the middle (evidence)
            t = t[:max_len - 24] + t[-24:]
        ids.append(t)
    return ids


def label_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer(l, add_special_tokens=False)["input_ids"][0] for l in LABELS]
    assert len(set(ids)) == 3, f"label first-tokens collide: {ids}"
    return ids


def collate(batch_ids: list[list[int]], pad_id: int, device):
    import torch
    width = max(map(len, batch_ids))
    inp, att = [], []
    for ids in batch_ids:  # LEFT pad so the last position is always the next-token position
        pad = width - len(ids)
        inp.append([pad_id] * pad + ids)
        att.append([0] * pad + [1] * len(ids))
    inp = torch.tensor(inp, device=device)
    att = torch.tensor(att, device=device)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return inp, att, pos


def last_logits(model, inp, att, pos):
    """Run backbone and apply lm_head ONLY at the final position (saves memory/time)."""
    inner = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    hidden = inner.model(input_ids=inp, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
    return inner.lm_head(hidden).float()


def predict(model, tokenizer, rows: list[dict], batch_size: int = 16, bias: tuple = (0.0, 0.0, 0.0)):
    """Return (predicted labels, probabilities over the 3 label tokens)."""
    import torch
    ids = encode(tokenizer, rows)
    lab = torch.tensor(label_token_ids(tokenizer), device=next(model.parameters()).device)
    order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
    probs = [None] * len(rows)
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), batch_size):
            chunk = order[s:s + batch_size]
            inp, att, pos = collate([ids[i] for i in chunk], pad_id, lab.device)
            logits = last_logits(model, inp, att, pos)[:, lab] + torch.tensor(bias, device=lab.device)
            p = torch.softmax(logits, -1).cpu().tolist()
            for i, pi in zip(chunk, p):
                probs[i] = pi
    preds = [LABELS[max(range(3), key=lambda k: p[k])] for p in probs]
    return preds, probs


def metrics(preds: list[str], golds: list[str]) -> dict:
    per, f1s = {}, []
    for l in LABELS:
        tp = sum(p == l and g == l for p, g in zip(preds, golds))
        fp = sum(p == l and g != l for p, g in zip(preds, golds))
        fn = sum(p != l and g == l for p, g in zip(preds, golds))
        pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        per[l] = {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)
    conf = {g: {p: sum(pp == p and gg == g for pp, gg in zip(preds, golds)) for p in LABELS} for g in LABELS}
    return {"n": len(golds), "accuracy": round(sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds)), 4),
            "macro_f1": round(sum(f1s) / 3, 4),
            "macro_precision": round(sum(v["precision"] for v in per.values()) / 3, 4),
            "macro_recall": round(sum(v["recall"] for v in per.values()) / 3, 4),
            "per_class": per, "confusion": conf}


def load_model(adapter: str | None, train: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def sample(rows, n, seed=0):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows
