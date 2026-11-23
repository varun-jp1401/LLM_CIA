# Risk-Aware Medical Answer Verification

A research prototype for studying how a language model can answer medical information questions, retrieve supporting text, check answer claims, and respond cautiously when evidence is insufficient.

> **Research demo only.** This project is not a medical device, has not been clinically validated, and must not be used to diagnose, treat, or make medication decisions.

## Current status

### Implemented

- Qwen3-1.7B generates draft answers.
- MedQuAD XML is parsed as individual question–answer pairs; empty answers are skipped.
- BM25-style lexical retrieval ranks MedQuAD passages, with simple question-intent and disease-subtype adjustments.
- Answers are split into claims and scored with a separate `cross-encoder/nli-deberta-v3-small` entailment model.
- A simple keyword risk screen flags certain medication, dosage, emergency, and diagnosis statements.
- The prototype attempts a grounded repair for some lower-risk claims, checks repaired text again, and abstains on selected high-risk cases.
- A Streamlit interface displays the answer, claim labels, evidence sources, and retrieved passages.
- The `llm` Conda environment was created locally with Python 3.11. On the original machine, PyTorch 2.11.0+cu128 detected an RTX 5050 Laptop GPU.

### Still to do

- **LoRA/QLoRA fine-tuning has not been run.** Training examples for claim verification, repair, and abstention have not yet been prepared. The app currently uses the un-fine-tuned Qwen3-1.7B model.
- **PubMedQA has not been connected to the app or evaluation.** The checked-in `data/PubMedQA/data/ori_pqal.json` and `test_ground_truth.json` are available for a later evaluation step.
- The PPT's intended PubMedBERT embeddings, FAISS, hybrid retrieval/RRF, and dedicated clinical NLI verifier are not implemented. Current retrieval uses BM25-style lexical matching; the independent NLI model is a general-purpose screening model.
- No complete comparison has been run for baseline Qwen, Qwen with retrieval, and the full verification/repair pipeline.
- Clinical expert review, metric calibration, and systematic safety evaluation remain outstanding.

## Project phases

| Phase | Status |
|---|---|
| Review proposal, set up environment, obtain datasets and Qwen model | Done |
| Build initial Streamlit generation/retrieval/verification prototype | Done |
| Improve MedQuAD parsing and question-intent retrieval; check sample questions | Prototype done; retrieval and NLI still need broader evaluation |
| Prepare task-specific examples and run Qwen3-1.7B LoRA/QLoRA fine-tuning | Not started |
| Integrate PubMedQA and compare the three PPT configurations | Not started |
| Final evaluation, documentation, and presentation update | In progress |

## Repository layout

```text
app.py                     Streamlit interface
pipeline.py                Qwen generation, MedQuAD retrieval, NLI checks, repair/abstention
requirements.txt           Python packages
data/MedQuAD/              MedQuAD dataset and its license/readme
data/PubMedQA/             PubMedQA source repository and included labeled data
LLM_CIA_1.pptx              Project proposal and implementation plan
```

## Setup on Windows

The original development environment used Conda and CUDA 12.8. These commands create a compatible environment and install a CUDA build of PyTorch before the other packages:

```powershell
conda create -n llm python=3.11 -y
conda activate llm
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

If the CUDA wheel is incompatible with your NVIDIA driver or GPU, install the PyTorch build recommended by the official PyTorch selector. CPU inference is possible but can be slow. The original machine used an RTX 5050 Laptop GPU with 8 GB VRAM.

The model files are not stored in this repository. Download/cache both models once before launching the app:

```powershell
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B'); AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-small'); AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-small')"
```

Then run:

```powershell
python -m streamlit run app.py
```

The first launch can take a while while model weights load. After changing pipeline code or data, restart Streamlit so its cached pipeline and index are rebuilt.

## Data and attribution

- **MedQuAD** is used for retrieval. The included repository states CC BY 4.0; retain the attribution and review the supplied `data/MedQuAD/LICENSE.txt` and README before reuse or redistribution.
- **PubMedQA** is included but not yet used by the app. The source repository includes a license and dataset notes; review them and the source paper before reuse. The unlabeled and artificially generated subsets mentioned in the PPT were not downloaded.
- No separate PubMed/NIH live retrieval corpus has been built.

## How the prototype behaves

1. Qwen drafts a response.
2. The pipeline retrieves MedQuAD question–answer pairs for the user question.
3. The response is split into sentence-like claims and compared with retrieved answer sentences using the NLI model.
4. A rule-based keyword screen assigns a rough risk label. Some lower-risk unmatched answers may be regenerated using evidence; selected higher-risk cases abstain.
5. The UI shows the response, claim-level screening, and retrieved passages.

These are prototype heuristics. `potentially_supported` is not proof that a claim is true; `neutral` means the verifier could not establish support. Retrieval can return related but inapplicable passages, the NLI model can misclassify claims, and generated answers can contain errors. Keep the limitations visible in any demo or report.

## Suggested next work

1. Build and audit a small training set of structured verification-feedback and repair examples.
2. Run a small Qwen3-1.7B LoRA experiment on the available hardware, save the adapter separately, and compare it with the unfine-tuned model. Do not describe fine-tuning as complete until the run and checks finish.
3. Add a reproducible PubMedQA evaluation path and report clear metrics.
4. Compare the three PPT configurations: baseline Qwen, Qwen plus retrieval, and the full verification/repair pipeline.
5. Inspect errors and limitations; expert clinical review is needed before making safety claims.
