# Risk-Aware Medical Answer Verification

A research prototype for studying how a language model can answer medical information questions, retrieve supporting text, check answer claims, and respond cautiously when evidence is insufficient.

> **Research demo only.** This project is not a medical device, has not been clinically validated, and must not be used to diagnose, treat, or make medication decisions.

## Current status

### Implemented

- Qwen3-1.7B generates draft answers.
- MedQuAD XML is parsed as individual question–answer pairs; empty answers are skipped.
- BM25-style lexical retrieval ranks MedQuAD passages, with simple question-intent and disease-subtype adjustments.
- **Hybrid retrieval (new):** PubMedBERT dense embeddings (FAISS flat-IP index) fused with BM25 via Reciprocal Rank Fusion (RRF). Falls back gracefully to BM25-only when `faiss-cpu` or `sentence-transformers` are unavailable.
- Answers are split into claims and scored with a separate `cross-encoder/nli-deberta-v3-small` entailment model.
- A simple keyword risk screen flags certain medication, dosage, emergency, and diagnosis statements.
- The prototype attempts a grounded repair for some lower-risk claims, checks repaired text again, and abstains on selected high-risk cases.
- **PubMedQA integration (new):** `evaluate.py` loads labeled PubMedQA examples and runs a reproducible three-way comparison (baseline / retrieval / full pipeline). Falls back to 10 synthetic examples if the data files are absent.
- **LoRA fine-tuning script (new):** `finetune.py` prepares 11 seed examples (claim verification, grounded repair, abstention) and can run a Qwen3-1.7B LoRA/QLoRA fine-tuning experiment. `--dry-run` validates data on CPU with no model download.
- A Streamlit interface (four tabs: Q&A, Evaluation, Fine-tuning, About) displays the answer, claim labels, evidence sources, retrieved passages, evaluation results, and fine-tuning status.
- The `llm` Conda environment was created locally with Python 3.11. On the original machine, PyTorch 2.11.0+cu128 detected an RTX 5050 Laptop GPU.

### Still to do

- **LoRA/QLoRA fine-tuning has not been run.** Training examples for claim verification, repair, and abstention have been prepared in `data/finetune_examples.json` (seed set). A full training run with a GPU session and a larger annotated dataset is needed.
- **Dataset download:** `data/MedQuAD/` and `data/PubMedQA/` are currently empty. Run `python download_data.py` to download both from their public GitHub repositories.
- No complete comparison has been run yet. Run `python evaluate.py --configs baseline retrieval full --n 50` after downloading PubMedQA to generate `results/eval_results.json`.
- Clinical expert review, metric calibration, and systematic safety evaluation remain outstanding.

## Project phases

| Phase | Status |
|---|---|
| Review proposal, set up environment, obtain datasets and Qwen model | Done |
| Build initial Streamlit generation/retrieval/verification prototype | Done |
| Improve MedQuAD parsing and question-intent retrieval; check sample questions | Prototype done; retrieval and NLI still need broader evaluation |
| Add PubMedBERT dense retrieval + FAISS + RRF hybrid | **Done** |
| Integrate PubMedQA and implement three-way comparison evaluator | **Done** |
| Prepare task-specific examples and run Qwen3-1.7B LoRA/QLoRA fine-tuning | Script ready, seed data prepared; **GPU training run not yet completed** |
| Final evaluation, documentation, and presentation update | In progress |

## Repository layout

```text
app.py                     Streamlit interface (4 tabs: Q&A, Evaluation, Fine-tuning, About)
pipeline.py                Qwen generation, MedQuAD retrieval, NLI checks, repair/abstention
retrieval.py               Hybrid BM25 + PubMedBERT/FAISS retrieval with RRF fusion (new)
evaluate.py                Three-way PubMedQA evaluation: baseline / retrieval / full (new)
finetune.py                LoRA fine-tuning script for Qwen3-1.7B (new)
download_data.py           Downloads MedQuAD and PubMedQA from GitHub (new)
requirements.txt           Python packages (updated with faiss-cpu, sentence-transformers, peft, trl, etc.)
data/MedQuAD/              MedQuAD dataset (run download_data.py to populate)
data/PubMedQA/             PubMedQA dataset (run download_data.py to populate)
data/finetune_examples.json  Seed fine-tuning examples (11 examples, auto-generated)
results/                   Evaluation output JSONs (created by evaluate.py)
models/                    LoRA adapter checkpoints (created by finetune.py)
LLM_CIA_1.pptx             Project proposal and implementation plan
```

## Setup on Windows

The original development environment used Conda and CUDA 12.8. The venv at `venv/` uses Python 3.10 (CPU torch).

```powershell
# Using the existing venv
c:\LLM\LLM_CIA\venv\Scripts\pip.exe install -r requirements.txt
```

For CUDA support (required for fine-tuning and faster inference):

```powershell
conda create -n llm python=3.11 -y
conda activate llm
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

### Download datasets

```powershell
python download_data.py
```

### Download model weights (once)

```powershell
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B'); AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-small'); AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-small')"
```

### Run the app

```powershell
python -m streamlit run app.py
```

### Run evaluation (after downloading PubMedQA)

```powershell
python evaluate.py --configs baseline retrieval full --n 50
```

### LoRA fine-tuning

```powershell
# CPU dry-run (no GPU needed)
python finetune.py --dry-run

# Full training (GPU required, ≥ 6 GB VRAM)
python finetune.py --epochs 3 --batch-size 2 --lora-r 16

# Merge adapter (optional)
python finetune.py --merge --adapter models/qwen3-1.7b-lora
```

## How the prototype behaves

1. Qwen drafts a response.
2. The pipeline retrieves MedQuAD (and optionally PubMedQA) passages using BM25 or hybrid RRF retrieval.
3. The response is split into sentence-like claims and compared with retrieved answer sentences using the NLI model.
4. A rule-based keyword screen assigns a rough risk label. Some lower-risk unmatched answers may be regenerated using evidence; selected higher-risk cases abstain.
5. The UI shows the response, claim-level screening, and retrieved passages.

These are prototype heuristics. `potentially_supported` is not proof that a claim is true; `neutral` means the verifier could not establish support. Retrieval can return related but inapplicable passages, the NLI model can misclassify claims, and generated answers can contain errors. Keep the limitations visible in any demo or report.

## Data and attribution

- **MedQuAD** is used for retrieval. The included repository states CC BY 4.0; retain the attribution and review the supplied `data/MedQuAD/LICENSE.txt` and README before reuse or redistribution.
- **PubMedQA** is used for evaluation. The source repository includes a license and dataset notes; review them and the source paper before reuse. The unlabeled and artificially generated subsets mentioned in the PPT were not downloaded.
- No separate PubMed/NIH live retrieval corpus has been built.

## Suggested next work

1. Run `python download_data.py` to populate the data directories, then `python evaluate.py` to get baseline metrics.
2. Augment `data/finetune_examples.json` with ~50–100 high-quality annotated examples per task.
3. Run `python finetune.py --epochs 3 --batch-size 2 --lora-r 16` on a GPU session; save the adapter separately.
4. Compare fine-tuned vs. un-fine-tuned model on the PubMedQA evaluation split.
5. Inspect errors and limitations; expert clinical review is needed before making safety claims.
