"""Few-shot prompted base Qwen3-1.7B (no adapter): 6 exemplars (1 real + 1 synthetic per class) from the TRAINING sets,
same restricted three-way scoring; dumps test predictions as tag `fewshot`."""
import json
from pathlib import Path
import torch
import claim_common as C

tok_model = C.load_model(None)
model, tok = tok_model
hv = C.load_healthver("train"); syn = C.load_synthetic("train")


def pick(rows, label, max_words=45):
    for r in rows:
        if r["label"] == label and len(r["evidence"].split()) <= max_words and len(r["claim"].split()) <= 25:
            return r


shots = []
for lab in ("entailment", "neutral", "contradiction"):
    shots.append(pick(hv, lab)); shots.append(pick(syn, lab, 60))
shots = [shots[i] for i in (0, 3, 4, 1, 2, 5)]  # interleave classes/sources
lab_ids = torch.tensor(C.label_token_ids(tok), device="cuda")
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def encode(rows):
    out = []
    for r in rows:
        msgs = []
        for s in shots:
            msgs += [{"role": "user", "content": C.build_user_text(s)}, {"role": "assistant", "content": s["label"]}]
        msgs.append({"role": "user", "content": C.build_user_text(r)})
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out.append(tok(txt, add_special_tokens=False)["input_ids"])
    return out


sets = {"medquad_test_iid": C.load_synthetic("test_iid"), "medquad_test_ood": C.load_synthetic("test_ood"), "healthver_test": C.load_healthver("test")}
res = {}
model.eval()
for name, rows in sets.items():
    ids = encode(rows); order = sorted(range(len(rows)), key=lambda i: len(ids[i])); probs = [None] * len(rows)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), 8):
            ch = order[s:s + 8]
            inp, att, pos = C.collate([ids[i] for i in ch], pad, lab_ids.device)
            p = torch.softmax(C.last_logits(model, inp, att, pos)[:, lab_ids], -1).cpu().tolist()
            for i, pi in zip(ch, p):
                probs[i] = pi
    preds = [C.LABELS[max(range(3), key=lambda k: p[k])] for p in probs]
    m = C.metrics(preds, [r["label"] for r in rows]); print(name, m["accuracy"], m["macro_f1"], flush=True)
    res[name] = [{"gold": r["label"], "pred": p, "probs": pr, "src": r["src"]} for r, p, pr in zip(rows, preds, probs)]
Path("results/predictions_fewshot.json").write_text(json.dumps(res), encoding="utf-8")
