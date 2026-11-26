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
import random
import re
import xml.etree.ElementTree as ET
import json
import sys
from pathlib import Path

DATA_FILE   = Path(__file__).resolve().parent.parent / "data" / "finetune_examples.json"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3-1.7b-lora"
BASE_MODEL  = "Qwen/Qwen3-1.7B"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PUBMEDQA_DIR = DATA_DIR / "PubMedQA" / "data"
SPLIT_DIR = DATA_DIR / "finetune_splits"


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


def _normalize_question(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _pubmedqa_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.items() if isinstance(data, dict) else enumerate(data)
    rows = []
    for uid, item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("QUESTION", item.get("question", ""))).strip()
        label = str(item.get("final_decision", item.get("answer", ""))).strip().lower()
        contexts = item.get("CONTEXTS", item.get("context", []))
        context = " ".join(map(str, contexts)) if isinstance(contexts, list) else str(contexts)
        if q and context and label in {"yes", "no", "maybe"}:
            rows.append({"id": str(uid), "question": q, "context": context, "label": label})
    return rows


def _stratified_split(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    result = {name: [] for name in ("train", "validation", "test")}
    for label in ("yes", "no", "maybe"):
        group = [row for row in rows if row["label"] == label]
        rng.shuffle(group)
        n_train, n_val = int(len(group) * .75), int(len(group) * .10)
        result["train"].extend(group[:n_train])
        result["validation"].extend(group[n_train:n_train + n_val])
        result["test"].extend(group[n_train + n_val:])
    for group in result.values():
        rng.shuffle(group)
    return result


def _load_medquad_train(excluded_questions: set[str]) -> list[dict]:
    """Create a separate reference-answer task from MedQuAD, train partition only."""
    root_dir = DATA_DIR / "MedQuAD"
    out = []
    for path in sorted(root_dir.rglob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        for pair in root.findall(".//QAPair") or [root]:
            qs = pair.findall(".//Question")
            ans = pair.findall(".//Answer")
            question = " ".join(" ".join(t.strip() for t in node.itertext() if t.strip()) for node in qs).strip()
            answer = " ".join(" ".join(t.strip() for t in node.itertext() if t.strip()) for node in ans).strip()
            key = _normalize_question(question)
            if not question or len(answer) < 40 or key in excluded_questions:
                continue
            out.append({
                "task": "medical_qa",
                "instruction": "Answer the medical information question using only the reference answer. Preserve uncertainty and keep the response concise.",
                "input": f"Question: {question}",
                "output": answer[:400],
                "question": question,
                "source": str(path.relative_to(root_dir)),
            })
    return out


def _split_medquad(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed + 1)
    unique = {}
    for row in rows:
        unique.setdefault(_normalize_question(row["question"]), row)
    rows = list(unique.values())
    rng.shuffle(rows)
    n_train, n_val = int(len(rows) * .75), int(len(rows) * .10)
    return {"train": rows[:n_train], "validation": rows[n_train:n_train+n_val], "test": rows[n_train+n_val:]}


def _pubmedqa_examples(rows: list[dict]) -> list[dict]:
    return [
        {"task": "pubmedqa", "instruction": "Answer the biomedical research question using the abstract evidence. Output exactly one label: yes, no, or maybe.",
        "input": f"Question: {row['question']}\nEvidence: {row['context'][:1200]}", "output": row["label"],
         "question": row["question"], "id": row["id"]}
        for row in rows
    ]


def prepare_splits(seed: int = 42, train_limit_per_task: int = 0, medquad_limit: int = 250) -> dict[str, list[dict]]:
    source = PUBMEDQA_DIR / "ori_pqal.json"
    if not source.exists():
        raise FileNotFoundError(source)
    pubmed_rows = _pubmedqa_rows(source)
    if not pubmed_rows:
        raise ValueError("No labeled PubMedQA records found.")
    pubmed_splits = _stratified_split(pubmed_rows, seed)
    normalize = lambda part: {_normalize_question(row["question"]) for row in part}
    qsets = {name: normalize(part) for name, part in pubmed_splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = qsets[left] & qsets[right]
        if overlap:
            raise ValueError(f"Question leakage between {left} and {right}: {len(overlap)} duplicates")

    rng = random.Random(seed)
    # Use the complete PubMedQA train partition by default. Optional limits
    # remain available for quick smoke runs, with deterministic stratification.
    pubmed_groups = {label: [r for r in pubmed_splits["train"] if r["label"] == label] for label in ("yes", "no", "maybe")}
    if train_limit_per_task <= 0 or train_limit_per_task >= len(pubmed_splits["train"]):
        selected_pubmed = list(pubmed_splits["train"])
    else:
        selected_pubmed = []
        total_pubmed_train = len(pubmed_splits["train"])
        expected = {label: train_limit_per_task * len(group) / total_pubmed_train for label, group in pubmed_groups.items()}
        allocations = {label: int(count) for label, count in expected.items()}
        for label in sorted(expected, key=lambda key: expected[key] - allocations[key], reverse=True)[:max(0, train_limit_per_task-sum(allocations.values()))]:
            allocations[label] += 1
        for label, group in pubmed_groups.items():
            selected_pubmed.extend(rng.sample(group, min(len(group), allocations[label])))
    train_pubmed = _pubmedqa_examples(selected_pubmed)
    medquad = _load_medquad_train(qsets["train"] | qsets["validation"] | qsets["test"])
    medquad_available = len(medquad)
    medquad_splits = _split_medquad(medquad, seed)
    medquad_sets = {name: {_normalize_question(row["question"]) for row in bucket} for name, bucket in medquad_splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if medquad_sets[left] & medquad_sets[right]:
            raise ValueError(f"MedQuAD question leakage between {left} and {right} partitions.")
    pubmed_all_questions = qsets["train"] | qsets["validation"] | qsets["test"]
    for name in medquad_splits:
        medquad_splits[name] = [r for r in medquad_splits[name]
                                if _normalize_question(r["question"]) not in pubmed_all_questions]
    medquad_train_available = len(medquad_splits["train"])
    rng.shuffle(medquad_splits["train"])
    medquad_train_used = medquad_splits["train"][:max(0, medquad_limit)]
    splits = {
        "train": train_pubmed + medquad_train_used,
        "validation": _pubmedqa_examples(pubmed_splits["validation"]) + medquad_splits["validation"],
        "test": _pubmedqa_examples(pubmed_splits["test"]) + medquad_splits["test"],
    }
    full_train_partition = _pubmedqa_examples(pubmed_splits["train"]) + medquad_splits["train"]
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "train.json").write_text(json.dumps(full_train_partition, indent=2, ensure_ascii=False), encoding="utf-8")
    (SPLIT_DIR / "train_sample.json").write_text(json.dumps(splits["train"], indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ("validation", "test"):
        (SPLIT_DIR / f"{name}.json").write_text(json.dumps(splits[name], indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "seed": seed, "split_ratios": {"train": .75, "validation": .10, "test": .15},
        "pubmedqa_total": len(pubmed_rows),
        "pubmedqa_counts": {name: {label: sum(r["label"] == label for r in part) for label in ("yes", "no", "maybe")}
                            for name, part in pubmed_splits.items()},
        "pubmedqa_train_used": {label: sum(row["label"] == label for row in selected_pubmed) for label in ("yes", "no", "maybe")},
        "medquad_available": medquad_available,
        "medquad_counts": {name: len(bucket) for name, bucket in medquad_splits.items()},
        "medquad_train_available": medquad_train_available,
        "medquad_train_used": len(medquad_train_used),
        "train_limit_per_task": train_limit_per_task,
        "medquad_train_limit": medquad_limit,
        "training_policy": "all PubMedQA train rows by default; capped MedQuAD train rows are task-balanced",
        "train_sample_size": len(splits["train"]),
        "question_overlap_counts": {"train_validation": len(qsets["train"] & qsets["validation"]),
                                     "train_test": len(qsets["train"] & qsets["test"]),
                                     "validation_test": len(qsets["validation"] & qsets["test"])},
    }
    (SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return splits


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

def _make_response_collator(tokenizer, max_length: int = 512):
    """Apply Qwen's chat template and mask prompt tokens from the causal LM loss."""
    import torch

    class ChatResponseCollator:
        def __call__(self, features):
            ids_batch, labels_batch = [], []
            for item in features:
                user_text = f"{item['instruction']}\n\n{item['input']}"
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if hasattr(prompt_ids, "input_ids"):
                    prompt_ids = prompt_ids.input_ids
                elif isinstance(prompt_ids, dict):
                    prompt_ids = prompt_ids["input_ids"]
                if hasattr(prompt_ids, "tolist"):
                    prompt_ids = prompt_ids.tolist()
                completion_ids = tokenizer(item["output"], add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    completion_ids = completion_ids + [tokenizer.eos_token_id]
                full_ids = (prompt_ids + completion_ids)[:max_length]
                prompt_len = min(len(prompt_ids), len(full_ids))
                labels = [-100] * prompt_len + full_ids[prompt_len:]
                ids_batch.append(full_ids)
                labels_batch.append(labels)
            width = max(map(len, ids_batch))
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            padded_ids, padded_labels, masks = [], [], []
            for ids, labels in zip(ids_batch, labels_batch):
                pad = width - len(ids)
                padded_ids.append(ids + [pad_id] * pad)
                padded_labels.append(labels + [-100] * pad)
                masks.append([1] * len(ids) + [0] * pad)
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    return ChatResponseCollator()


def _calibrate_choice_bias(model, tokenizer, rows: list[dict], output_dir: Path) -> dict:
    """Tune a small maybe-label logit offset on PubMedQA validation only."""
    import torch

    labels = ("yes", "no", "maybe")
    label_ids = {label: tokenizer(label, add_special_tokens=False)["input_ids"][0] for label in labels}
    observed = []
    model.eval()
    for row in rows:
        prompt = f"{row['instruction']}\n\n{row['input']}"
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        batch = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**batch).logits[0, -1, :]
            scores = logits[torch.tensor([label_ids[label] for label in labels], device=logits.device)]
        observed.append((row["output"].strip().lower(), scores.float().cpu().tolist()))

    def macro_f1(bias: float) -> float:
        predictions = [labels[max(range(3), key=lambda i: scores[i] + (bias if labels[i] == "maybe" else 0.0))]
                       for _, scores in observed]
        f1s = []
        golds = [gold for gold, _ in observed]
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, golds))
            fp = sum(p == label and g != label for p, g in zip(predictions, golds))
            fn = sum(p != label and g == label for p, g in zip(predictions, golds))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1s.append(2 * precision * recall / max(1e-9, precision + recall))
        return sum(f1s) / len(f1s)

    # A narrow, predeclared search avoids fitting independent thresholds for
    # each class on the small validation set. Ties favor the least adjustment.
    candidates = (-0.4, -0.2, 0.0, 0.2, 0.4)
    bias = max(candidates, key=lambda value: (macro_f1(value), -abs(value)))
    result = {"yes": 0.0, "no": 0.0, "maybe": bias,
              "selection": "validation macro F1; maybe-only offset; grid -0.4..0.4"}
    (output_dir / "choice_bias.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[finetune] Validation-calibrated maybe logit bias: {bias:+.1f} (macro F1 {macro_f1(bias):.4f})")
    return result


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

    splits = prepare_splits(seed=args.seed, train_limit_per_task=args.train_limit_per_task,
                            medquad_limit=args.medquad_limit)
    train_rows = splits["train"]
    validation_rows = splits["validation"]
    test_rows = splits["test"]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Train, validation, and test partitions must all be non-empty.")
    print(f"[finetune] Train: {len(train_rows)} ({len([r for r in train_rows if r.get('task') == 'pubmedqa'])} PubMedQA + MedQuAD)")
    print(f"[finetune] Validation: {len(validation_rows)} total / {sum(r.get('task') == 'pubmedqa' for r in validation_rows)} PubMedQA")
    print(f"[finetune] Test held out: {len(test_rows)} total / {sum(r.get('task') == 'pubmedqa' for r in test_rows)} PubMedQA; not used for training or checkpoint selection")

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

    output_dir = Path(args.output_dir)
    if (output_dir / "adapter_model.safetensors").exists():
        raise FileExistsError(f"Adapter already exists at {output_dir}; choose a new --output-dir to preserve this run.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- response-only collator (works with any trl version) ---
    # Explicit response-only causal LM collator. This avoids SFTTrainer silently
    # replacing the previously configured collator and masking only the prompt.
    collator = _make_response_collator(tokenizer)
    from transformers import Trainer, TrainingArguments

    train_ds = Dataset.from_list(train_rows)
    # Select checkpoints against the PubMedQA task; retain the MedQuAD
    # validation partition for independent generative QA evaluation.
    val_ds = Dataset.from_list([r for r in validation_rows if r.get("task") == "pubmedqa"])

    common_kwargs = dict(
        output_dir=str(output_dir),
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
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=0,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    training_args = TrainingArguments(**common_kwargs)
    from torch.utils.data import WeightedRandomSampler

    class TaskBalancedTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            rows = eval_dataset if eval_dataset is not None else self.eval_dataset
            label_ids = {label: tokenizer(label, add_special_tokens=False)["input_ids"][0]
                         for label in ("yes", "no", "maybe")}
            golds, preds = [], []
            self.model.eval()
            for row in rows:
                prompt = f"{row['instruction']}\n\n{row['input']}"
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                batch = tokenizer(rendered, return_tensors="pt").to(self.model.device)
                with torch.inference_mode():
                    logits = self.model(**batch).logits[0, -1, :]
                    scores = logits[torch.tensor(list(label_ids.values()), device=logits.device)]
                preds.append(list(label_ids)[int(scores.argmax())])
                golds.append(row["output"].strip().lower())
            class_f1 = []
            for label in ("yes", "no", "maybe"):
                tp = sum(p == label and g == label for p, g in zip(preds, golds))
                fp = sum(p == label and g != label for p, g in zip(preds, golds))
                fn = sum(p != label and g == label for p, g in zip(preds, golds))
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                class_f1.append(2 * precision * recall / max(1e-9, precision + recall))
            metrics[f"{metric_key_prefix}_macro_f1"] = sum(class_f1) / len(class_f1)
            self.log({f"{metric_key_prefix}_macro_f1": metrics[f"{metric_key_prefix}_macro_f1"]})
            return metrics

        def _get_train_sampler(self, train_dataset=None):
            ds = train_dataset if train_dataset is not None else self.train_dataset
            # Balance labels at the example level, so rare "maybe" cases are
            # represented as often as yes/no during optimizer updates.
            pubmed_labels = [row.get("output", "").strip().lower() for row in ds if row.get("task") == "pubmedqa"]
            pubmed_counts = {label: pubmed_labels.count(label) for label in set(pubmed_labels)}
            medquad_n = sum(row.get("task") == "medical_qa" for row in ds)
            weights = []
            for row in ds:
                if row.get("task") == "pubmedqa":
                    # Half the draws go to PubMedQA, split evenly across labels.
                    weights.append(0.5 / (len(pubmed_counts) * pubmed_counts[row["output"].strip().lower()]))
                else:
                    # Half the draws go to distinct MedQuAD examples.
                    weights.append(0.5 / max(1, medquad_n))
            return WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)

    trainer = TaskBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("[finetune] Starting training …")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _calibrate_choice_bias(trainer.model, tokenizer, list(val_ds), output_dir)
    print(f"[finetune] Adapter saved to {output_dir}")


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit-per-task", type=int, default=0,
                        help="Optional cap on PubMedQA rows; 0 uses the full PubMedQA train split")
    parser.add_argument("--medquad-limit", type=int, default=250,
                        help="Number of seeded MedQuAD train examples to mix in; set 0 to disable")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "models" / "qwen3-1.7b-lora-pubmedqa"),
                        help="Write new adapter here; defaults to a separate directory to preserve the old adapter")
    parser.add_argument("--dry-run",    action="store_true", help="Validate data only, no model loading")
    parser.add_argument("--merge",      action="store_true", help="Merge LoRA adapter into base model")
    parser.add_argument("--adapter",    default=str(ADAPTER_DIR), help="Path to LoRA adapter for --merge")
    args = parser.parse_args()

    if args.dry_run:
        splits = prepare_splits(seed=args.seed, train_limit_per_task=args.train_limit_per_task,
                                medquad_limit=args.medquad_limit)
        print(json.dumps({"train": len(splits["train"]), "validation": len(splits["validation"]), "test": len(splits["test"]),
                          "manifest": json.loads((SPLIT_DIR / "manifest.json").read_text(encoding="utf-8"))}, indent=2))
    elif args.merge:
        merge(args.adapter)
    else:
        train(args)
