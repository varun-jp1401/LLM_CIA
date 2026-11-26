"""
LoRA fine-tune Qwen3-1.7B for 3-way claim verification (entailment / neutral / contradiction).

Training mix: real HealthVer train rows + synthetic MedQuAD claims (topic-disjoint from val/test).
Loss: cross-entropy on the first answer token (full vocab) -> still a normal generative model,
      but scoring only the 3 label tokens at inference gives clean, calibratable decisions.
Model selection: best macro-F1 on (HealthVer dev + MedQuAD val); early stopping. Test sets are
never touched here.

    python train_claim_verifier.py --n-real 4000 --n-synth 2500 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import claim_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=4000, help="HealthVer train rows (max 5005)")
    ap.add_argument("--n-synth", type=int, default=2500, help="synthetic MedQuAD train rows")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ppt-config", action="store_true", help="r=8, q_proj+v_proj only (as in the slides)")
    ap.add_argument("--eval-every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--val-real", type=int, default=450)
    ap.add_argument("--val-synth", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop early")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(C.ROOT / "models" / "qwen3-1.7b-lora-claimverify"))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    real = C.sample(C.load_healthver("train"), args.n_real, args.seed) if args.n_real > 0 else []
    synth = C.sample(C.load_synthetic("train"), args.n_synth, args.seed) if args.n_synth > 0 else []
    train_rows = real + synth
    random.shuffle(train_rows)
    val_rows = C.sample(C.load_healthver("dev"), args.val_real, 1) + C.sample(C.load_synthetic("val"), args.val_synth, 1)
    print(f"[train] {len(train_rows)} train ({len(real)} real + {len(synth)} synthetic); {len(val_rows)} val")

    model, tok = C.load_model(None)
    from peft import LoraConfig, get_peft_model
    targets = ["q_proj", "v_proj"] if args.ppt_config else \
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    r = 8 if args.ppt_config else args.rank
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.1, target_modules=targets,
                                             bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    ids = C.encode(tok, train_rows)
    lab_ids = C.label_token_ids(tok)
    y = [lab_ids[C.LABELS.index(r_["label"])] for r_ in train_rows]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = next(model.parameters()).device

    eff_bs = args.micro_bs * args.accum
    steps_per_epoch = math.ceil(len(train_rows) / eff_bs)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warm = max(5, int(0.05 * total_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    print(f"[train] {total_steps} optimizer steps (eff batch {eff_bs}); eval every {args.eval_every}")

    def evaluate():
        preds, _ = C.predict(model, tok, val_rows, batch_size=16)
        return C.metrics(preds, [r_["label"] for r_ in val_rows])

    best, bad, step, t0, log = -1.0, 0, 0, time.time(), []
    order: list[int] = []
    run_loss = 0.0
    model.train()
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if len(order) < args.micro_bs:
                order = list(range(len(train_rows)))
                random.shuffle(order)
            chunk, order = order[:args.micro_bs], order[args.micro_bs:]
            inp, att, pos = C.collate([ids[i] for i in chunk], pad_id, dev)
            tgt = torch.tensor([y[i] for i in chunk], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = C.last_logits(model, inp, att, pos)
            loss = F.cross_entropy(logits, tgt) / args.accum
            loss.backward()
            run_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 10 == 0:
            print(f"[train] step {step}/{total_steps} loss {run_loss / 10:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"elapsed {(time.time() - t0) / 60:.1f}m", flush=True)
            run_loss = 0.0
        if step % args.eval_every == 0 or step == total_steps:
            m = evaluate()
            model.train()
            log.append({"step": step, **{k: m[k] for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall")}})
            print(f"[val] step {step}: acc {m['accuracy']} macroF1 {m['macro_f1']} "
                  f"P {m['macro_precision']} R {m['macro_recall']}", flush=True)
            if m["macro_f1"] > best:
                best, bad = m["macro_f1"], 0
                model.save_pretrained(str(out))
                tok.save_pretrained(str(out))
                (out / "best_val.json").write_text(json.dumps({"step": step, **m}, indent=2), encoding="utf-8")
                print(f"[val] new best {best:.4f} -> saved {out}", flush=True)
            else:
                bad += 1
                if bad >= args.patience:
                    print("[train] early stopping (no val improvement)")
                    break
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2), encoding="utf-8")
    print(f"[train] done. best val macro-F1 {best:.4f}. adapter in {out}")


if __name__ == "__main__":
    main()
