"""
LoRA / QLoRA fine-tuning script for Qwen3-1.7B.
Updated for trl>=1.0 / transformers>=5.0 compatibility.

Tasks trained together (instruction-following format):
  1. Claim verification  — given a claim + evidence, output entail / neutral / contradict
  2. Grounded repair     — given an unverified draft + evidence, produce a corrected answer
  3. Abstention          — given a high-risk claim with no evidence, output the abstention string

Usage:
    python finetune.py --dry-run                        # validate data only, no GPU needed
    python finetune.py --epochs 3 --batch-size 2 --lora-r 16
    python finetune.py --merge --adapter models/qwen3-1.7b-lora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_FILE   = Path(__file__).parent / "data" / "finetune_examples.json"
ADAPTER_DIR = Path(__file__).parent / "models" / "qwen3-1.7b-lora"
BASE_MODEL  = "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# Seed data (fallback if finetune_examples.json is missing)
# ---------------------------------------------------------------------------

SEED_EXAMPLES = [
    {
        "task": "claim_verification",
        "instruction": "Classify whether the evidence supports, is neutral to, or contradicts the claim. Reply with one word: entailment, neutral, or contradiction.",
        "input": "Claim: Metformin is first-line treatment for type 2 diabetes.\nEvidence: Metformin is recommended as the first-line pharmacological therapy for type 2 diabetes by most major guidelines.",
        "output": "entailment",
    },
    {
        "task": "claim_verification",
        "instruction": "Classify whether the evidence supports, is neutral to, or contradicts the claim. Reply with one word: entailment, neutral, or contradiction.",
        "input": "Claim: Ibuprofen is safe during pregnancy.\nEvidence: NSAIDs including ibuprofen are generally contraindicated in pregnancy, especially in the third trimester, due to risks of premature closure of the ductus arteriosus.",
        "output": "contradiction",
    },
    {
        "task": "grounded_repair",
        "instruction": "Revise the draft answer to be accurate and grounded in the evidence. Keep the response concise and cite the evidence.",
        "input": "Question: What is the first-line treatment for hypertension?\nDraft: The first-line treatment for hypertension is typically a beta-blocker.\nEvidence: Current guidelines recommend thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers as first-line antihypertensive agents; beta-blockers are generally second-line.",
        "output": "Current clinical guidelines recommend thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers as first-line antihypertensive agents. Beta-blockers are typically considered second-line therapy.",
    },
    {
        "task": "abstention",
        "instruction": "Decide whether to answer or abstain. If the claim involves high-risk medication or diagnosis advice and there is no supporting evidence, output the abstention message.",
        "input": "Claim: You should double your prescribed dose of warfarin if your symptoms worsen.\nEvidence: [none]",
        "output": "I can't verify this potentially high-risk medication claim against available evidence. Please consult a qualified healthcare professional before changing any prescribed medication dose.",
    },
]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(ex: dict) -> str:
    return (
        f"### Instruction\n{ex['instruction']}\n\n"
        f"### Input\n{ex['input']}\n\n"
        f"### Response\n{ex['output']}"
    )


def _ensure_data() -> list[dict]:
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            print(f"[finetune] Loaded {len(data)} examples from {DATA_FILE}")
            return data
        except Exception as e:
            print(f"[finetune] Could not read {DATA_FILE}: {e}; using seed data.")
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(SEED_EXAMPLES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[finetune] Saved {len(SEED_EXAMPLES)} seed examples to {DATA_FILE}.")
    return SEED_EXAMPLES


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_response_collator(tokenizer, response_template: str = "### Response\n"):
    """
    Build a data collator that only computes loss on the response portion of each
    prompt, masking out instruction + input tokens.

    Tries trl's built-in class first (available up to trl ~0.9), then falls back
    to a minimal custom implementation compatible with trl 1.x.
    """
    # --- try built-in (trl <= 0.9) ---
    for import_path in [
        ("trl", "DataCollatorForCompletionOnlyLM"),
        ("trl.trainer.utils", "DataCollatorForCompletionOnlyLM"),
    ]:
        try:
            mod = __import__(import_path[0], fromlist=[import_path[1]])
            cls = getattr(mod, import_path[1])
            ids = tokenizer.encode(response_template, add_special_tokens=False)
            print(f"[finetune] Using {import_path[0]}.{import_path[1]} collator.")
            return cls(ids, tokenizer=tokenizer)
        except (ImportError, AttributeError):
            continue

    # --- custom fallback for trl >= 1.0 ---
    print("[finetune] DataCollatorForCompletionOnlyLM not found in trl; using custom response-only collator.")
    import torch
    from transformers import DataCollatorForSeq2Seq

    response_ids = tokenizer.encode(response_template, add_special_tokens=False)

    class _ResponseOnlyCollator:
        """Tokenize full prompts and mask everything before '### Response\\n'."""
        def __init__(self, tokenizer, max_length: int = 512):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, features):
            texts   = [f["text"] if isinstance(f, dict) else f for f in features]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            labels    = input_ids.clone()

            # Mask tokens that come before the response template
            tmpl_len = len(response_ids)
            for i, row in enumerate(input_ids):
                row_list = row.tolist()
                # Find last occurrence of the response template token sequence
                found = -1
                for j in range(len(row_list) - tmpl_len, -1, -1):
                    if row_list[j: j + tmpl_len] == response_ids:
                        found = j
                        break
                if found != -1:
                    labels[i, : found + tmpl_len] = -100  # mask instruction + template
                else:
                    labels[i, :] = -100  # template not found — mask everything

            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100
            encoded["labels"] = labels
            return encoded

    return _ResponseOnlyCollator(tokenizer)


def train(args):
    # --- imports ---
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"[finetune] Required package missing: {e}")
        sys.exit(1)

    try:
        from trl import SFTTrainer, SFTConfig
        USE_SFTCONFIG = True
    except ImportError:
        from trl import SFTTrainer
        USE_SFTCONFIG = False

    # --- data ---
    examples = _ensure_data()
    if len(examples) < 4:
        print("[finetune] WARNING: very few training examples. Run generate_finetune_data.py first.")
    texts = [_build_prompt(ex) for ex in examples]

    dataset = Dataset.from_dict({"text": texts})
    test_size = max(1, int(len(texts) * 0.1)) / len(texts)
    split = dataset.train_test_split(test_size=test_size, seed=42)
    print(f"[finetune] Train: {len(split['train'])}  Eval: {len(split['test'])}")

    # --- model ---
    print(f"[finetune] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune] Device: {device}  dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.enable_input_require_grads()

    # --- LoRA ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # --- response-only collator (works with any trl version) ---
    collator = _make_response_collator(tokenizer)

    # --- training args ---
    common_kwargs = dict(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        warmup_steps=5,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        save_total_limit=2,
    )

    if USE_SFTCONFIG:
        training_args = SFTConfig(
            **common_kwargs,
            max_length=512,
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            args=training_args,
        )
    else:
        from transformers import TrainingArguments
        training_args = TrainingArguments(**common_kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            dataset_text_field="text",
            max_seq_length=512,
            args=training_args,
        )

    print("[finetune] Starting training …")
    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"[finetune] Adapter saved to {ADAPTER_DIR}")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(adapter_path: str):
    """Merge LoRA adapter into base model and save as full model."""
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        print(f"[finetune] {e}"); sys.exit(1)

    print(f"[finetune] Loading base model and adapter from {adapter_path} …")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    out_dir = Path(adapter_path).parent / "qwen3-1.7b-merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[finetune] Merged model saved to {out_dir}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run():
    examples = _ensure_data()
    print(f"\n[finetune] Dry run: {len(examples)} examples. Sample prompts:")
    for ex in examples[:2]:
        print("\n" + "=" * 60)
        print(_build_prompt(ex))
    print("\n[finetune] Data validation OK — no model loaded (dry run).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen3-1.7B for medical claim verification"
    )
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-r",     type=int, default=16, help="LoRA rank (default 16)")
    parser.add_argument("--dry-run",    action="store_true", help="Validate data only, no model loading")
    parser.add_argument("--merge",      action="store_true", help="Merge LoRA adapter into base model")
    parser.add_argument("--adapter",    default=str(ADAPTER_DIR), help="Path to LoRA adapter for --merge")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    elif args.merge:
        merge(args.adapter)
    else:
        train(args)
