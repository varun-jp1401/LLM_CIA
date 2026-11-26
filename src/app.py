"""
Medical Answer Verification — Streamlit UI
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from pipeline import MedicalPipeline

st.set_page_config(
    page_title="Medical Answer Verification",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — monospace, violet/black scheme, centered tabs, bigger font
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font: JetBrains Mono ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

/* ── Global font override ────────────────────────────────── */
html, body, .stApp {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 15px;
    letter-spacing: -0.01em;
}

/* Explicit font styling for standard text and markdown elements */
.stMarkdown, .stText, p, li, h1, h2, h3, h4, h5, h6,
.stTextInput input, .stTextArea textarea,
.stSelectbox label, .stSlider label, .stCheckbox label,
[data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
}

/* Expander layout & clean icon rendering */
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

[data-testid="stExpander"] summary p {
    font-family: 'JetBrains Mono', monospace !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Streamlit Material Symbols icon protection */
[class*="e1vmumty"],
[data-testid*="Icon"],
[data-testid="stExpanderStepChevron"] *,
[data-testid="stExpander"] summary span:first-child,
[data-testid="stExpander"] summary svg {
    font-family: 'Material Symbols Rounded' !important;
    font-feature-settings: 'liga' 1 !important;
    -webkit-font-feature-settings: 'liga' 1 !important;
    letter-spacing: normal !important;
    font-style: normal !important;
    font-weight: normal !important;
    line-height: 1 !important;
    text-transform: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    max-width: 1.5rem !important;
    overflow: hidden !important;
}

h1 { font-size: 2rem !important; font-weight: 600 !important; }
h2 { font-size: 1.5rem !important; font-weight: 500 !important; }
h3 { font-size: 1.15rem !important; font-weight: 500 !important; }

/* ── Light mode ──────────────────────────────────────────── */
@media (prefers-color-scheme: light) {
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #5B21B6 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #faf5ff !important;
        border-right: 1px solid #e9d5ff;
    }
    [data-testid="stSidebar"] * { color: #5B21B6 !important; }

    /* Headings */
    h1, h2, h3 { color: #3b0764 !important; }

    /* Paragraphs and body text */
    p, span, div, label { color: #4c1d95 !important; }

    /* Inputs */
    .stTextArea textarea, .stTextInput input {
        background-color: #faf5ff !important;
        border: 1px solid #c4b5fd !important;
        color: #3b0764 !important;
        border-radius: 6px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 2px #ede9fe !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid*="primary"] {
        background-color: #7C3AED !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #5B21B6 !important;
    }

    /* Secondary buttons */
    .stButton > button {
        background-color: #ede9fe !important;
        color: #5B21B6 !important;
        border: 1px solid #c4b5fd !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background-color: #ddd6fe !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        border: 1px solid #e9d5ff !important;
        border-radius: 8px !important;
        background: #faf5ff !important;
    }

    /* Metric / info boxes */
    [data-testid="stAlert"] {
        border-radius: 6px !important;
    }

    /* Tabs active underline */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #7C3AED !important;
        color: #5B21B6 !important;
    }

    /* Code blocks */
    code, pre, .stCode { background: #f3e8ff !important; color: #3b0764 !important; }
}

/* ── Dark mode ───────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #1a1a1a;
    }
    [data-testid="stSidebar"] * { color: #e5e5e5 !important; }

    h1, h2, h3 { color: #ffffff !important; }
    p, span, div, label { color: #e5e5e5 !important; }

    /* Inputs */
    .stTextArea textarea, .stTextInput input {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 2px #1e0a3a !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #7C3AED !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #6D28D9 !important;
    }

    /* Secondary buttons */
    .stButton > button {
        background-color: #111111 !important;
        color: #e5e5e5 !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        border-color: #7C3AED !important;
        color: #c4b5fd !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        border: 1px solid #1f1f1f !important;
        border-radius: 8px !important;
        background: #0d0d0d !important;
    }

    /* Tabs active underline */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #7C3AED !important;
        color: #c4b5fd !important;
    }

    /* Code blocks */
    code, pre, .stCode {
        background: #111111 !important;
        color: #c4b5fd !important;
    }

    /* Dataframe / table */
    [data-testid="stDataFrame"] { background: #0d0d0d !important; }
}

/* ── Tab bar — centred + clean ───────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center !important;
    gap: 2rem !important;
    border-bottom: 1px solid #d8b4fe40 !important;
    padding-bottom: 0 !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 0 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #9ca3af !important;
    transition: color 0.2s, border-color 0.2s !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #7C3AED !important;
    border-bottom-color: #7C3AED40 !important;
    background: transparent !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Dividers ────────────────────────────────────────────── */
hr { border-color: #7C3AED30 !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #7C3AED50; border-radius: 2px; }

/* ── Remove Streamlit branding ───────────────────────────── */
[data-testid="stToolbar"], footer { display: none !important; }

/* ── Badge/tag pills ─────────────────────────────────────── */
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px !important;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid #c4b5fd;
    color: #7C3AED;
    background: transparent;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Qwen3-1.7B, claim-verification adapter and NLI model …")
def get_pipeline():
    """One shared model: base Qwen3-1.7B + two switchable LoRA adapters (older Q&A adapter, claim verifier)."""
    return MedicalPipeline(adapter_path="models/qwen3-1.7b-lora",
                           claim_adapter_path="models/qwen3-1.7b-lora-claimverify")


@st.cache_resource(show_spinner="Loading hybrid retrieval index …")
def get_retriever():
    try:
        from retrieval import HybridRetriever
        return HybridRetriever(use_dense=True)
    except ImportError:
        return None
    except Exception as e:
        st.warning(f"Hybrid retriever unavailable ({e}). Falling back to BM25.")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Settings")
use_lora = st.sidebar.checkbox(
    "Use older Q&A LoRA adapter for answer drafting",
    value=False,
    help="Affects the Q & A tab only. models/qwen3-1.7b-lora (early prototype) tends to refuse to answer, "
         "so the base model drafts answers by default. The Claim check tab always uses the claim-verification adapter.",
)
use_hybrid = st.sidebar.checkbox(
    "Hybrid retrieval (BM25 + PubMedBERT/FAISS)",
    value=True,
    help="Uses dense FAISS index cached at data/faiss_index.pkl. Blends lexical BM25 with PubMedBERT.",
)
top_k = st.sidebar.slider("Retrieved passages (k)", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype · Qwen3-1.7B · Not for clinical use")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_claim, tab_qa, tab_eval, tab_finetune, tab_about = st.tabs(
    ["Claim check", "Q & A", "Evaluation", "Fine-tuning", "About"]
)

DEMO_CLAIMS = [
    "Glaucoma can damage the optic nerve and lead to vision loss.",
    "Antibiotics are an effective treatment for viral infections such as the common cold.",
    "Asthma is caused by a bacterial infection and is contagious.",
    "Aspirin is recommended for all children with viral fevers.",
    "Cataracts cause clouding of the lens of the eye.",
    "Parkinson disease is a contagious disease.",
]
VERDICT_UI = {"supported": ("SUPPORTED", "success"), "contradicted": ("CONTRADICTED", "error"),
              "conflicting": ("CONFLICTING EVIDENCE", "warning"), "not_enough_evidence": ("NOT ENOUGH EVIDENCE", "info")}

# =============================================================================
# TAB 0 — Claim check (the model evaluated in the paper)
# =============================================================================
with tab_claim:
    st.title("Medical Claim Verification")
    st.caption("Evidence retrieval · Qwen3-1.7B + claim-verification LoRA (entailment / neutral / contradiction) · DeBERTa NLI baseline")
    st.markdown("---")
    st.markdown("Type a medical claim, or pick an example. The claim is checked against retrieved MedQuAD evidence.")
    example = st.selectbox("Example claims", ["(write your own)"] + DEMO_CLAIMS, key="claim_example")
    claim_text = st.text_area("Claim", value="" if example == "(write your own)" else example,
                              key=f"claim_text_{example}", height=80,
                              placeholder="e.g. Antibiotics are an effective treatment for the common cold.")

    if st.button("Verify claim", type="primary", disabled=not claim_text.strip()):
        from claim_verifier import check_claim
        try:
            pipe = get_pipeline()
            retriever = get_retriever() if use_hybrid else None
            with st.spinner("Retrieving evidence and verifying the claim …"):
                res = check_claim(pipe, claim_text.strip(), retriever=retriever, k=top_k)
            label, kind = VERDICT_UI[res["verdict"]]
            getattr(st, kind)(f"**{label}** — {res['explanation']}")
            st.caption(f"Verifier: {res['backend']}")

            if res.get("nli"):
                n = res["nli"]
                st.markdown(f"**DeBERTa NLI baseline** (zero-shot, the deployed pipeline's verifier): "
                            f"best-passage label `{n['label']}`, entailment score {n['entailment_score']:.2f}")

            st.markdown("#### Evidence windows scored by the LoRA verifier")
            for i, w in enumerate(res["windows"], 1):
                pr = w["probs"]
                with st.expander(f"Window {i} — {w['pred'].upper()} ({pr[w['pred']]:.0%}) · overlap {w['overlap']:.2f}", expanded=i == 1):
                    st.write(w["text"])
                    st.caption(f"entailment {pr['entailment']:.2f} · neutral {pr['neutral']:.2f} · contradiction {pr['contradiction']:.2f} · source: {w['source']}")
            st.caption("Per-window labels come from the evaluated model. The rule that combines windows into one verdict "
                       "(≥ 0.50 confidence, ≥ 0.35 claim-word overlap) is a demo heuristic and was not evaluated. "
                       "On real claims (HealthVer) the verifier reaches macro-F1 ≈ 0.76. Research prototype — not for clinical use.")
        except Exception as e:
            st.error(f"Verification error: {e}")

# =============================================================================
# TAB 1 — Q&A
# =============================================================================
with tab_qa:
    st.title("Medical Answer Verification")
    st.caption("Qwen3-1.7B · evidence retrieval · NLI claim verification · selective repair")
    st.markdown("---")

    question = st.text_area(
        "Enter a medical question",
        placeholder="e.g. What are the common causes of hypertension?",
        height=110,
    )

    if st.button("Generate and verify", type="primary", disabled=not question.strip()):
        pipe = get_pipeline()
        pipe.use_generation_adapter = use_lora
        retriever = get_retriever() if use_hybrid else None

        with st.spinner("Generating answer, retrieving evidence, verifying claims …"):
            try:
                result = pipe.run(question, retriever=retriever)
                result["retriever"] = "hybrid_rrf" if retriever is not None else "bm25"
                retriever_label = "Hybrid RRF" if result.get("retriever") == "hybrid_rrf" else "BM25"

                # --- Final answer ---
                st.markdown("#### Final response")
                st.caption(f"Retrieval mode: {retriever_label} | Model: {'LoRA Fine-Tuned' if use_lora else 'Base Qwen3-1.7B'}")
                st.write(result["answer"])

                if result["abstained"]:
                    st.warning("Insufficient evidence for a safe answer. The pipeline abstained from repair.")

                st.markdown("---")

                # --- Verification details ---
                st.markdown("#### Claim verification")
                st.caption(result["note"])

                for i, claim in enumerate(result["claims"], 1):
                    risk_label = "HIGH RISK" if claim["risk"] == "high" else "MEDIUM RISK"
                    status_label = "SUPPORTED" if claim["status"] == "potentially_supported" else "UNVERIFIED"
                    with st.expander(f"Claim {i} — {claim['claim'][:85]}…"):
                        st.write(claim["claim"])
                        st.markdown(
                            f'<span class="tag">{risk_label}</span>'
                            f'<span class="tag">{status_label}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"NLI label: {claim['verifier_label']} "
                            f"({claim['confidence']:.0%} confidence) · "
                            f"Source: {claim['evidence_source'] or 'none'}"
                        )

                st.markdown("---")

                # --- Retrieved evidence ---
                with st.expander(f"Retrieved evidence — {len(result['evidence'])} passages"):
                    for item in result["evidence"]:
                        retriever_tag = item.get("retriever", "bm25")
                        corpus_tag = item.get("corpus", "MedQuAD")
                        st.markdown(
                            f"**{item['source']}** · score `{item['score']:.4f}` "
                            f"· `{retriever_tag}` · `{corpus_tag}`"
                        )
                        st.write(item["text"][:500])
                        st.divider()

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.info("Check README setup instructions and model / data paths.")

# =============================================================================
# TAB 2 — Evaluation
# =============================================================================
with tab_eval:
    st.title("PubMedQA Evaluation")
    st.caption("Compare three pipeline configurations on labeled biomedical questions.")
    st.markdown("---")

    st.markdown(
        "Three configurations are benchmarked:\n\n"
        "1. **Baseline** — Qwen3-1.7B generation only, no retrieval\n"
        "2. **Retrieval** — Qwen3-1.7B with hybrid evidence retrieval\n"
        "3. **Full** — Retrieval + NLI verification + repair / abstention"
    )

    results_path = Path(__file__).resolve().parent.parent / "results" / "eval_results.json"

    col1, col2 = st.columns([2, 1])
    with col1:
        n_samples = st.slider("Number of evaluation samples", 5, 100, 20)
        configs = st.multiselect(
            "Configurations to run",
            ["baseline", "retrieval", "full"],
            default=["baseline", "retrieval", "full"],
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_eval = st.button("Run evaluation >>", type="primary")

    if results_path.exists():
        with st.expander("Load previously saved results"):
            try:
                saved = json.loads(results_path.read_text(encoding="utf-8"))
                import pandas as pd
                rows = [
                    {
                        "Config": cfg,
                        "Accuracy": r.get("accuracy", "-"),
                        "Abstention rate": r.get("abstention_rate", "-"),
                        "Avg time (s)": r.get("avg_time_s", "-"),
                        "N": r.get("n", "-"),
                    }
                    for cfg, r in saved.items()
                ]
                st.dataframe(pd.DataFrame(rows).set_index("Config").astype(str), width="stretch")
                with st.expander("Full JSON"):
                    st.json(saved)
            except Exception:
                pass

    if run_eval and configs:
        pipe = get_pipeline()
        pipe.use_generation_adapter = use_lora
        retriever = get_retriever() if use_hybrid else None
        from evaluate import Evaluator

        evaluator = Evaluator(pipe, retriever=retriever, max_samples=n_samples)
        all_results = {}
        progress = st.progress(0.0, text="Starting evaluation …")

        for i, cfg in enumerate(configs):
            st.info(f"Running: {cfg} …")
            if cfg == "baseline":
                r = evaluator.run_baseline()
            elif cfg == "retrieval":
                r = evaluator.run_retrieval()
            else:
                r = evaluator.run_full()
            all_results[cfg] = r
            progress.progress((i + 1) / len(configs), text=f"Completed: {cfg}")

        st.success("Evaluation complete.")

        import pandas as pd
        rows = [
            {
                "Config": cfg,
                "Accuracy": r.get("accuracy"),
                "Abstention rate": r.get("abstention_rate"),
                "Avg NLI score": r.get("avg_entailment_score", "-"),
                "Avg retrieval score": r.get("avg_retrieval_score", "-"),
                "Avg time (s)": r.get("avg_time_s"),
                "N": r.get("n"),
            }
            for cfg, r in all_results.items()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Config").astype(str), width="stretch")

        with st.expander("Full JSON results"):
            st.json(all_results)

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        st.caption(f"Saved to {results_path}")

# =============================================================================
# TAB 3 — Fine-tuning
# =============================================================================
with tab_finetune:
    st.title("LoRA Fine-tuning")
    st.caption("Task-specific fine-tuning of Qwen3-1.7B using Low-Rank Adaptation.")
    st.markdown("---")

    adapter_dir = Path(__file__).resolve().parent.parent / "models" / "qwen3-1.7b-lora"
    finetune_data = Path(__file__).resolve().parent.parent / "data" / "finetune_examples.json"

    st.markdown(
        "Three training tasks are supported:\n\n"
        "- **Claim verification** — classify evidence as entailment / neutral / contradiction\n"
        "- **Grounded repair** — rewrite a draft answer grounded in retrieved evidence\n"
        "- **Abstention** — detect and refuse high-risk unverifiable claims"
    )

    st.markdown("#### Terminal commands")
    st.code(
        "# Step 1 — validate data (CPU-safe dry run)\n"
        "python finetune.py --dry-run\n\n"
        "# Step 2 — full fine-tuning (requires GPU with >= 6 GB VRAM)\n"
        "python finetune.py --epochs 3 --batch-size 2 --lora-r 16\n\n"
        "# Step 3 — optional: merge adapter into full model\n"
        "python finetune.py --merge --adapter models/qwen3-1.7b-lora",
        language="bash",
    )

    st.markdown("---")
    st.markdown("#### Training data")

    if finetune_data.exists():
        try:
            examples = json.loads(finetune_data.read_text(encoding="utf-8"))
            st.caption(f"{len(examples)} examples · {finetune_data}")

            task_counts: dict[str, int] = {}
            for ex in examples:
                k = ex.get("task", "unknown")
                task_counts[k] = task_counts.get(k, 0) + 1
            st.bar_chart(task_counts)

            with st.expander("Add a training example"):
                new_task = st.selectbox("Task", ["claim_verification", "grounded_repair", "abstention"])
                new_instruction = st.text_area("Instruction", key="new_instr")
                new_input = st.text_area("Input", key="new_inp")
                new_output = st.text_area("Output", key="new_out")
                if st.button("Add example"):
                    if new_instruction and new_input and new_output:
                        examples.append({
                            "task": new_task,
                            "instruction": new_instruction,
                            "input": new_input,
                            "output": new_output,
                        })
                        finetune_data.write_text(
                            json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        st.success(f"Added example #{len(examples)}")
                        st.rerun()
        except Exception as e:
            st.error(f"Could not read training data: {e}")

    st.markdown("---")
    st.markdown("#### Adapter status")
    if adapter_dir.exists() and any(adapter_dir.iterdir()):
        adapter_files = list(adapter_dir.iterdir())
        st.success(f"LoRA adapter active at `{adapter_dir}`")
        st.caption(f"Files: {', '.join(f.name for f in adapter_files[:8])}")
    else:
        st.warning("No LoRA adapter found.")

# =============================================================================
# TAB 4 — About
# =============================================================================
with tab_about:
    st.title("About this prototype")
    st.caption("Risk-aware medical answer verification — research prototype, not for clinical use.")
    st.markdown("---")

    st.markdown("""
**Medical Answer Verification** is a research prototype studying how a language model can produce
safer, evidence-grounded responses to medical information queries.

The pipeline has four stages:

1. **Generation** — Qwen3-1.7B (Base or Fine-Tuned LoRA) drafts an answer given the user question.
2. **Retrieval** — MedQuAD and PubMedQA passages are retrieved via BM25 lexical matching or a hybrid
   BM25 + PubMedBERT/FAISS dense index with Reciprocal Rank Fusion (RRF).
3. **Verification** — The draft is split into individual claims and each is scored against
   retrieved evidence using batched GPU `cross-encoder/nli-deberta-v3-small`.
4. **Repair or abstention** — Low-risk unverified claims are regenerated using evidence as context.
   High-risk claims with no support trigger a full abstention.
""")

    st.markdown("---")
    st.markdown("#### Pipeline architecture")
    st.code("""
User question
     |
     v
Qwen3-1.7B (Base / LoRA)  -->  Draft answer
     |
     +-->  Hybrid Retrieval (BM25 + PubMedBERT/FAISS)  -->  Evidence passages
     |
     +-->  Claim splitter  -->  Batched NLI verifier  -->  Risk classifier
     |
     +-->  Repair engine / Abstention
                   |
                   v
            Final response
""", language="text")

    st.markdown("---")
    st.markdown("#### Implementation status")

    st.markdown("""
| Component | Status |
|---|---|
| Qwen3-1.7B generation | Done |
| MedQuAD BM25 retrieval | Done |
| NLI claim verification (Batched GPU) | Done |
| Repair and abstention | Done |
| PubMedBERT dense retrieval (FAISS) | Done |
| PubMedQA integration and evaluation | Done |
| LoRA fine-tuning script | Done |
| LoRA adapter trained | **Done** (`models/qwen3-1.7b-lora`, eval acc 71.16%) |
| Three-way comparison benchmark | Done (Run via Evaluation tab) |
""")

    st.markdown("---")
    with st.expander("How to interpret labels"):
        st.markdown("""
| Label | Meaning |
|---|---|
| `potentially_supported` | NLI entailment score >= 0.70. Automated screening only; not proof of correctness. |
| `unverified` | NLI could not confirm the claim against retrieved evidence. |
| `neutral` | No strong entailment or contradiction detected. |
| `high risk` | Claim contains medication dosage, diagnosis, or emergency-care keywords. |
| `abstained` | Pipeline declined to answer due to high-risk unverifiable claims. |
""")

    st.caption("Research demo only. Not a medical device. Not clinically validated.")
