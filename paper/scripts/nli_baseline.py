"""Zero-shot DeBERTa-v3-small NLI cross-encoder (the project's independent verifier) on the same test sets."""
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import claim_common as C
name = "cross-encoder/nli-deberta-v3-small"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()
id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
print(id2)
tomap = {"contradiction": "contradiction", "entailment": "entailment", "neutral": "neutral"}
import sys
VAL = len(sys.argv) > 1 and sys.argv[1] == "val"
sets = {"medquad_val": C.load_synthetic("val"), "healthver_dev": C.load_healthver("dev")} if VAL else {"medquad_test_iid": C.load_synthetic("test_iid"), "medquad_test_ood": C.load_synthetic("test_ood"),
        "healthver_test": C.load_healthver("test")}
out = {}
torch.set_num_threads(8)
for n, rows in sets.items():
    preds, probs = [], []
    for s in range(0, len(rows), 32):
        ch = rows[s:s + 32]
        b = tok([r["evidence"] for r in ch], [r["claim"] for r in ch], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            p = torch.softmax(model(**b).logits, -1)
        for row in p:
            d = {tomap[id2[i]]: float(row[i]) for i in range(3)}
            probs.append([d[l] for l in C.LABELS]); preds.append(max(d, key=d.get))
    gold = [r["label"] for r in rows]
    m = C.metrics(preds, gold)
    print(n, m["accuracy"], m["macro_f1"], m["macro_precision"], m["macro_recall"], flush=True)
    out[n] = [{"gold": g, "pred": p, "probs": pr, "src": r["src"]} for g, p, pr, r in zip(gold, preds, probs, rows)]
Path("results/predictions_nli_deberta_val.json" if VAL else "results/predictions_nli_deberta.json").write_text(json.dumps(out), encoding="utf-8")
