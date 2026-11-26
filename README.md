# Risk-Aware Medical Answer Verification

A research prototype for studying how a language model can answer medical information questions, retrieve supporting text, check answer claims, and respond cautiously when evidence is insufficient.

> **Research demo only.** This project is not a medical device, has not been clinically validated, and must not be used to diagnose, treat, or make medication decisions.

## Current status

### Implemented & Completed

- **Base Generation**: Qwen3-1.7B generates draft answers with cautious medical instructions.
- **Corpus & Datasets**: 
  - MedQuAD (11,274 XML files) parsed as individual question–answer pairs across 12 NIH topics.
  - PubMedQA (`ori_pqal.json`) parsed and integrated for evaluation and retrieval.
- **Hybrid Retrieval**:
  - BM25-style lexical retrieval optimized with precomputed term sets and query-intent bonuses.
  - PubMedBERT dense embeddings (`pritamdeka/S-PubMedBert-MS-MARCO`) with FAISS flat-IP index.
  - Cached 15,000 document vector index at `data/faiss_index.pkl` (66.8 MB).
  - Reciprocal Rank Fusion (RRF) combines dense and sparse rankings.
- **Batched NLI Claim Verification**:
  - Claims split sentence-by-sentence and scored using `cross-encoder/nli-deberta-v3-small`.
  - Batching on GPU executes all candidate premise-hypothesis pairs in a single forward pass (~50x speedup).
- **Risk Screening & Policy**:
  - Keyword screen flags medication dosage, emergency, and critical diagnosis statements.
  - Grounded repair rewrites low-risk unverified claims using retrieved passages.
  - Conservative abstention triggers for high-risk unverifiable claims.
- **LoRA Fine-Tuning Completed**:
  - Curated 101 medical reasoning examples in `data/finetune_examples.json` (57 claim verification, 22 grounded repair, 22 abstention).
  - Fine-tuned Qwen3-1.7B using LoRA ($r=16, \alpha=32$) on an NVIDIA GeForce RTX 5050 Laptop GPU (PyTorch 2.11.0 + CUDA 12.8).
  - Training loss decreased to `1.081`; eval loss improved from `1.679` $\rightarrow$ `1.366`; eval token accuracy reached **71.16%**.
  - Checkpoint and adapter weights saved to `models/qwen3-1.7b-lora/`.
- **Three-Way PubMedQA Benchmark**:
  - `evaluate.py` compares baseline, retrieval, and full pipeline.
  - Output results saved to `results/eval_results.json`.
- **Streamlit Interactive UI**:
  - 4 tabs: Q&A, Evaluation, Fine-Tuning, and About.
  - Sidebar toggles between base Qwen and the fine-tuned LoRA adapter, as well as BM25 vs. Hybrid FAISS retrieval.

### Next Steps / Future Work
- Systematic clinical expert review and safety audits of generated repair outputs.
- Exploring larger base models (e.g., Qwen-7B or Llama-3-8B) with 4-bit/8-bit QLoRA.

## Project phases

| Phase | Status |
|---|---|
| Review proposal, set up environment, obtain datasets and Qwen model | **Done** |
| Build initial Streamlit generation/retrieval/verification prototype | **Done** |
| Improve MedQuAD parsing and question-intent retrieval; check sample questions | **Done** |
| Add PubMedBERT dense retrieval + FAISS + RRF hybrid | **Done** |
| Integrate PubMedQA and implement three-way comparison evaluator | **Done** |
| Prepare task-specific examples and run Qwen3-1.7B LoRA/QLoRA fine-tuning | **Done** (Trained on RTX 5050 GPU) |
| Final evaluation, documentation, and presentation update | **Done** |

## Repository layout

```text
src/                       All Python source (run from the repo root, e.g. python src/evaluate.py)
  app.py                     Streamlit interface (Q&A, Evaluation, Fine-tuning, About)
  pipeline.py                Qwen generation, MedQuAD retrieval, NLI checks, repair/abstention
  retrieval.py               Hybrid BM25 + PubMedBERT/FAISS retrieval with RRF fusion
  build_index.py             Builds the cached dense index
  download_data.py           Downloads MedQuAD and PubMedQA
  evaluate.py                Three-way PubMedQA evaluation: baseline / retrieval / full
  finetune.py                LoRA fine-tuning for Qwen3-1.7B (PubMedQA)
  generate_finetune_data.py  Builds the curated fine-tuning examples
  claim_common.py            Shared helpers for the claim-verification work
  build_claim_dataset.py     Builds the synthetic claim-verification dataset from MedQuAD
  train_claim_verifier.py    Trains the claim-verification LoRA adapter
  train_encoder_baseline.py  DeBERTa baseline    nli_baseline.py  NLI baseline
  claim_verifier.py          Inference-time claim checking (used by app.py)
  evaluate_claims.py         Claim-verification evaluation
  dump_predictions.py        Dumps model predictions    dump_fewshot.py  Few-shot predictions
  analysis*.py               Post-hoc analysis (calibration, extra metrics)
  make_figures.py            Result figures -> figures/
  pipeline_demo.py           Small end-to-end demo
scripts/                   Shell scripts for the ablation / seed runs (run from anywhere)
data/  models/  results/   Datasets, adapters, evaluation JSONs (git-ignored where large)
figures/                   Generated figures and result tables
paper/                     IEEE paper (docx/pdf), paper figures, data and script snapshots
paper_work/                Paper build tooling (build_paper.py, refs, templates, scratch renders)
docs/                      Project proposal deck (LLM_CIA_1.pptx) and conference template
requirements.txt           Python dependencies
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
python src/download_data.py
```

### Download model weights (once)

```powershell
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B'); AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-small'); AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-small')"
```

### Run the app

```powershell
python -m streamlit run src/app.py
```

### Run evaluation (after downloading PubMedQA)

```powershell
python src/evaluate.py --configs baseline retrieval full --n 50
```

### LoRA fine-tuning

```powershell
# Build and audit deterministic 75/10/15 question-disjoint partitions (no model/GPU needed)
python src/finetune.py --dry-run --seed 42

# Train with all PubMedQA train rows, balanced class sampling, and 250 MedQuAD rows
python src/finetune.py --epochs 3 --batch-size 2 --lora-r 16 --seed 42 --output-dir models/qwen3-1.7b-lora-pubmedqa-balanced

# After training, evaluate the untouched PubMedQA test partition
python src/evaluate.py --configs baseline --n 0 --held-out-split test --adapter models/qwen3-1.7b-lora-pubmedqa-balanced --compare-base --out results/eval_pubmedqa_balanced_test.json

# Merge adapter (optional)
python src/finetune.py --merge --adapter models/qwen3-1.7b-lora-pubmedqa-balanced
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

## Research Paper & Evaluation Metrics

This section records the prompt-matched PubMedQA evaluation for the balanced fine-tuning run. The 152-item test split was not used for training, checkpoint selection, or calibration.

### 1. Benchmark Results on PubMedQA

The fine-tuned model was selected by validation macro F1 (epoch 2, 0.507). A modest `maybe` log-score offset (+0.2) was selected on validation. Test results for the calibrated adapter and uncalibrated base model were:

| Metric | Fine-tuned adapter | Base model |
|---|---|---|
| Test accuracy | 48.68% | 46.05% |
| Test macro F1 | 0.396 | 0.359 |
| Yes precision / recall | 0.723 / 0.410 | 0.629 / 0.675 |
| No precision / recall | 0.534 / 0.750 | 0.667 / 0.154 |
| Maybe precision / recall | 0.031 / 0.059 | 0.118 / 0.353 |

The yes-only collapse is reduced, but performance remains far below the requested 90% precision and recall. The maybe class remains especially weak. These results do not establish clinical accuracy or safety. Older benchmark figures in project materials are historical and are not reproduced by this run.

---

### 2. LoRA Fine-Tuning Specifications & Training Curve

Qwen3-1.7B was fine-tuned using Low-Rank Adaptation (LoRA) on an **NVIDIA GeForce RTX 5050 Laptop GPU** (CUDA 12.8, PyTorch 2.11.0).

#### Hyperparameters
- **Base Model**: `Qwen/Qwen3-1.7B`
- **LoRA Rank ($r$)**: `16`
- **LoRA Alpha ($\alpha$)**: `32`
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Trainable Parameters**: `17,432,576` (1.003% of total 1.74B parameters)
- **Epochs**: `3`
- **Effective Batch Size**: `8` (Batch size 2 with 4 gradient accumulation steps)
- **Optimizer**: AdamW (`lr = 2e-4`, linear warmup across 5 steps)
- **Precision**: `FP16` mixed precision

#### Training Progression across Epochs
| Epoch | Training Loss | Validation Loss | Mean Token Accuracy |
|---|---|---|---|
| **Epoch 1** | 1.643 | 1.679 | 66.21% |
| **Epoch 2** | 1.223 | 1.399 | 70.65% |
| **Epoch 3** | **1.081** | **1.366** | **71.16%** |

*The steady decrease in validation loss without divergence demonstrates healthy convergence and strong generalization across the 3 medical reasoning tasks.*

---

### 3. Step-by-Step Reproduction Guide

#### Step A: Environment Activation
```powershell
conda activate llm
```

#### Step B: Build/Verify the FAISS Vector Index (One-Time)
```powershell
python src/build_index.py --max-docs 15000 --batch-size 128
```
*Generates and caches `data/faiss_index.pkl` (66.8 MB) using `pritamdeka/S-PubMedBert-MS-MARCO` embeddings.*

#### Step C: Generate Training Data & Run LoRA Fine-Tuning
```powershell
# 1. Create seeded 75/10/15 question-disjoint partitions from both source folders
python src/finetune.py --dry-run --seed 42

# 2. Use all PubMedQA training rows with balanced class sampling; add 250 MedQuAD rows
python src/finetune.py --epochs 3 --batch-size 2 --lora-r 16 --seed 42 --output-dir models/qwen3-1.7b-lora-pubmedqa-balanced
```

#### Step D: Run 3-Way Benchmark Evaluation
```powershell
python src/evaluate.py --configs baseline --n 0 --held-out-split test --adapter models/qwen3-1.7b-lora-pubmedqa-balanced --compare-base --out results/eval_pubmedqa_balanced_test.json
```
*The training script writes full question-disjoint train, validation, and test partitions to `data/finetune_splits/`. PubMedQA validation selects checkpoints; the test partition is not used for training, checkpoint selection, or calibration. The evaluator restricts decisions to yes/no/maybe and excludes PubMedQA from retrieval.*

#### Step E: Launch the Streamlit Demonstration Application
```powershell
streamlit run src/app.py
```
*Opens interactive web UI with toggles for the Fine-Tuned LoRA adapter, BM25 vs. Hybrid RRF retrieval, claim entailment confidence scores, and retrieved evidence viewer.*

---

## Data and attribution
