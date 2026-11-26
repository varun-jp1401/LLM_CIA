"""Fine-tune the DeBERTa-v3-small NLI cross-encoder (142M parameters) on the SAME training mixture, to test whether
in-domain supervision alone (without an LLM + LoRA) closes the gap. Test predictions dumped as tag `deberta_ft`."""
import json, math, random
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import claim_common as C

NAME = "cross-encoder/nli-deberta-v3-small"
random.seed(42); torch.manual_seed(42)
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME).cuda()
id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
lab2id = {v: k for k, v in id2.items()}               # contradiction/entailment/neutral -> id
train = C.sample(C.load_healthver("train"), 4000, 42) + C.sample(C.load_synthetic("train"), 2500, 42)
random.shuffle(train)
val = C.sample(C.load_healthver("dev"), 450, 1) + C.sample(C.load_synthetic("val"), 300, 1)


def batches(rows, bs, shuffle=False):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for s in range(0, len(idx), bs):
        ch = [rows[i] for i in idx[s:s + bs]]
        enc = tok([r["evidence"] for r in ch], [r["claim"] for r in ch], padding=True, truncation=True, max_length=512, return_tensors="pt")
        yield {k: v.cuda() for k, v in enc.items()}, torch.tensor([lab2id[r["label"]] for r in ch]).cuda(), ch


def predict(rows):
    model.eval(); probs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for enc, _, ch in batches(rows, 32):
            p = torch.softmax(model(**enc).logits.float(), -1).cpu()
            for row in p:
                probs.append([float(row[lab2id[l]]) for l in C.LABELS])
    return [C.LABELS[max(range(3), key=lambda k: p[k])] for p in probs], probs


EPOCHS, BS = 3, 16
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total = EPOCHS * math.ceil(len(train) / BS)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
best, best_state = -1, None
for ep in range(EPOCHS):
    model.train()
    for enc, y, _ in batches(train, BS, True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(**enc).logits.float(), y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
    pv, _ = predict(val); m = C.metrics(pv, [r["label"] for r in val])
    print(f"epoch {ep + 1}: val macro-F1 {m['macro_f1']} acc {m['accuracy']}", flush=True)
    if m["macro_f1"] > best:
        best = m["macro_f1"]; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
sets = C.load_eval_sets("test")
res = {}
for n, rows in sets.items():
    pr, probs = predict(rows); m = C.metrics(pr, [r["label"] for r in rows]); print(n, m["accuracy"], m["macro_f1"], flush=True)
    res[n] = [{"gold": r["label"], "pred": p, "probs": q, "src": r["src"]} for r, p, q in zip(rows, pr, probs)]
Path("results/predictions_deberta_ft.json").write_text(json.dumps(res), encoding="utf-8")
# validation predictions (for recalibration) as well
vsets = C.load_eval_sets("val")
vres = {}
for n, rows in vsets.items():
    pr, probs = predict(rows)
    vres[n] = [{"gold": r["label"], "pred": p, "probs": q, "src": r["src"]} for r, p, q in zip(rows, pr, probs)]
Path("results/predictions_deberta_ft_val.json").write_text(json.dumps(vres), encoding="utf-8")
