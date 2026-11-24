"""
LoRA / QLoRA fine-tuning script for Qwen3-1.7B.

Tasks trained together (instruction-following format):
  1. Claim verification  — given a claim + evidence passage, output entail / neutral / contradict
  2. Grounded repair     — given an unverified draft + evidence, produce a corrected answer
  3. Abstention          — given a high-risk claim with no evidence, output the abstention string

Training data:
  - If 'data/finetune_examples.json' exists, it is loaded directly.
  - Otherwise a small synthetic seed set is created and saved so the script is
    runnable immediately.  Run once to generate the file, then augment it with
    real examples before a production run.

Usage:
    python finetune.py --epochs 2 --batch-size 4 --lora-r 16
    python finetune.py --dry-run   # validate data only, no GPU needed

The adapter is saved to 'models/qwen3-1.7b-lora/'.
Merge it back with:
    python finetune.py --merge --adapter models/qwen3-1.7b-lora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "finetune_examples.json"
ADAPTER_DIR = Path(__file__).parent / "models" / "qwen3-1.7b-lora"
BASE_MODEL = "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# Synthetic seed data
# ---------------------------------------------------------------------------

SEED_EXAMPLES = [
    # --- Claim verification ---
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
        "task": "claim_verification",
        "instruction": "Classify whether the evidence supports, is neutral to, or contradicts the claim. Reply with one word: entailment, neutral, or contradiction.",
        "input": "Claim: Vitamin C can cure the common cold.\nEvidence: Regular supplementation with vitamin C does not prevent colds in the general population but may modestly reduce their duration.",
        "output": "contradiction",
    },
    {
        "task": "claim_verification",
        "instruction": "Classify whether the evidence supports, is neutral to, or contradicts the claim. Reply with one word: entailment, neutral, or contradiction.",
        "input": "Claim: Exercise improves cardiovascular health.\nEvidence: Aerobic exercise is associated with improved cardiovascular outcomes, reduced blood pressure, and lower risk of heart disease.",
        "output": "entailment",
    },
    {
        "task": "claim_verification",
        "instruction": "Classify whether the evidence supports, is neutral to, or contradicts the claim. Reply with one word: entailment, neutral, or contradiction.",
        "input": "Claim: Blood type determines personality.\nEvidence: There is no scientific evidence linking ABO blood type to personality traits.",
        "output": "contradiction",
    },
    # --- Grounded repair ---
    {
        "task": "grounded_repair",
        "instruction": "Revise the draft answer to be accurate and grounded in the evidence. Keep the response concise and cite the evidence.",
        "input": "Question: What is the first-line treatment for hypertension?\nDraft: The first-line treatment for hypertension is typically a beta-blocker.\nEvidence: Current guidelines recommend thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers as first-line antihypertensive agents; beta-blockers are generally second-line.",
        "output": "Current clinical guidelines recommend thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers as first-line antihypertensive agents. Beta-blockers are typically considered second-line therapy.",
    },
    {
        "task": "grounded_repair",
        "instruction": "Revise the draft answer to be accurate and grounded in the evidence. Keep the response concise and cite the evidence.",
        "input": "Question: Is the keto diet safe for people with type 1 diabetes?\nDraft: The keto diet is completely safe for type 1 diabetes and helps control blood sugar.\nEvidence: Very-low-carbohydrate diets in type 1 diabetes require careful medical supervision due to risks of hypoglycaemia and diabetic ketoacidosis; they are not universally recommended.",
        "output": "Very-low-carbohydrate (ketogenic) diets in type 1 diabetes carry risks including hypoglycaemia and diabetic ketoacidosis and require careful medical supervision. They are not universally recommended and should not be started without consulting a healthcare provider.",
    },
    {
        "task": "grounded_repair",
        "instruction": "Revise the draft answer to be accurate and grounded in the evidence. Keep the response concise and cite the evidence.",
        "input": "Question: Does coffee cause cancer?\nDraft: Coffee definitely causes cancer.\nEvidence: The World Health Organization removed coffee from its list of possible carcinogens in 2016; some evidence suggests coffee may reduce risk of certain cancers.",
        "output": "According to available evidence, coffee is not classified as a carcinogen; the WHO removed it from its list of possible carcinogens. Some studies suggest regular coffee consumption may be associated with a reduced risk of certain cancers.",
    },
    # --- Abstention ---
    {
        "task": "abstention",
        "instruction": "Decide whether to answer or abstain. If the claim involves high-risk medication or diagnosis advice and there is no supporting evidence, output the abstention message.",
        "input": "Claim: You should double your prescribed dose of warfarin if your symptoms worsen.\nEvidence: [none]",
        "output": "I can't verify this potentially high-risk medication claim against available evidence. Please consult a qualified healthcare professional before changing any prescribed medication dose.",
    },
    {
        "task": "abstention",
        "instruction": "Decide whether to answer or abstain. If the claim involves high-risk medication or diagnosis advice and there is no supporting evidence, output the abstention message.",
        "input": "Claim: Stopping antibiotics early when you feel better is safe.\nEvidence: [none]",
        "output": "I can't verify this claim against available evidence. Stopping antibiotics early can contribute to antibiotic resistance and treatment failure. Please consult a healthcare professional about your specific antibiotic course.",
    },
    {
        "task": "abstention",
        "instruction": "Decide whether to answer or abstain. If the claim involves high-risk medication or diagnosis advice and there is no supporting evidence, output the abstention message.",
        "input": "Claim: Taking acetaminophen and ibuprofen together is always safe.\nEvidence: [none]",
        "output": "I can't verify this claim against the available evidence. Combining acetaminophen and ibuprofen can be safe in some circumstances but requires careful attention to dosages and individual health conditions. Please consult a pharmacist or healthcare provider.",
    },
]


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

def train(args):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
        from datasets import Dataset
    except ImportError as e:
        print(f"[finetune] Required package missing: {e}\nInstall with: pip install peft trl datasets")
        sys.exit(1)

    examples = _ensure_data()
    texts = [_build_prompt(ex) for ex in examples]

    dataset = Dataset.from_dict({"text": texts})
    # Simple 90/10 train/val split
    split = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"[finetune] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        warmup_ratio=0.1,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    response_template = "### Response\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=512,
        data_collator=collator,
        args=training_args,
    )

    print("[finetune] Starting training …")
    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"[finetune] Adapter saved to {ADAPTER_DIR}")


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
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-1.7B for medical claim verification")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default 16)")
    parser.add_argument("--dry-run", action="store_true", help="Validate data only, no model loading")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", default=str(ADAPTER_DIR), help="Path to LoRA adapter for --merge")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    elif args.merge:
        merge(args.adapter)
    else:
        train(args)
