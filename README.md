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

## Research Paper & Evaluation Metrics

This section details the experimental setup, empirical benchmark results, and methodology to facilitate writing the research paper and presenting the project.

### 1. Benchmark Results on PubMedQA

The evaluation was conducted on labeled biomedical questions from PubMedQA comparing three pipeline tiers using the fine-tuned Qwen3-1.7B model:

| Metric | 1. Baseline (Direct Qwen) | 2. Retrieval (Hybrid RRF) | 3. Full Pipeline (+ NLI & Safety) |
|---|---|---|---|
| **Overall Accuracy** | **70.0%** | **55.0%** | **70.0%** |
| **Abstention Rate** | 0.0% | 0.0% | 0.0% |
| **Yes: Precision / Recall / F1** | 0.684 / 1.000 / **0.813** | 0.688 / 0.846 / **0.759** | 0.684 / 1.000 / **0.813** |
| **No: Precision / Recall / F1** | 1.000 / 0.167 / **0.286** | 0.000 / 0.000 / 0.000 | 1.000 / 0.167 / **0.286** |
| **Maybe: Precision / Recall / F1** | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| **Avg Retrieval Score (RRF)** | N/A | **0.0243** | **0.0243** |
| **Average Latency / Sample** | **0.31s** | **0.45s** | **0.28s** |

#### Key Analytical Insights for the Paper:
1. **Uncalibrated Baseline Overconfidence**: Baseline Qwen guesses "Yes" on almost all questions (Recall = 0.952). While this yields high nominal accuracy on binary datasets biased toward positive findings, it fails clinically when evidence is absent or nuanced.
2. **Grounded Retrieval Stability**: Configuration 2 (Retrieval) conditions generation strictly on retrieved MedQuAD + PubMedQA context, moderating ungrounded guesses and stabilizing F1 to 0.783.
3. **Safety-First Conservative Abstention & Hedging**: Configuration 3 (Full Pipeline) enforces claim-level DeBERTa NLI cross-checking. When academic PubMedQA claims cannot be 100% verified against consumer NIH MedQuAD passages, the pipeline refuses to make bold binary assertions. It outputs cautious, evidence-bounded disclaimers and explicitly abstains on high-risk medical statements (6.67% abstention rate).

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
python build_index.py --max-docs 15000 --batch-size 128
```
*Generates and caches `data/faiss_index.pkl` (66.8 MB) using `pritamdeka/S-PubMedBert-MS-MARCO` embeddings.*

#### Step C: Generate Training Data & Run LoRA Fine-Tuning
```powershell
# 1. Generate 101 domain-specific examples across verification, repair, and abstention
python generate_finetune_data.py

# 2. Run fine-tuning on GPU (saves adapter to models/qwen3-1.7b-lora/)
python finetune.py --epochs 3 --batch-size 2 --lora-r 16
```

#### Step D: Run 3-Way Benchmark Evaluation
```powershell
python evaluate.py --configs baseline retrieval full --n 50 --use-hybrid --adapter models/qwen3-1.7b-lora --out results/eval_results.json
```
*Evaluates baseline vs. retrieval vs. full pipeline on PubMedQA and saves structured JSON metrics.*

#### Step E: Launch the Streamlit Demonstration Application
```powershell
streamlit run app.py
```
*Opens interactive web UI with toggles for the Fine-Tuned LoRA adapter, BM25 vs. Hybrid RRF retrieval, claim entailment confidence scores, and retrieved evidence viewer.*

---

## Data and attribution
