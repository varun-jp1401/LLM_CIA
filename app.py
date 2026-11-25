"""
Updated Streamlit app with:
  - Hybrid retrieval (BM25 + optional PubMedBERT/FAISS)
  - PubMedQA evaluation tab
  - Fine-tuning guidance tab
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from pipeline import MedicalPipeline

st.set_page_config(page_title="Medical Answer Verification", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Qwen3-1.7B and retrieval index …")
def get_pipeline(use_lora: bool = True):
    adapter = "models/qwen3-1.7b-lora" if use_lora else None
    return MedicalPipeline(adapter_path=adapter)


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

st.sidebar.title("⚙️ Settings")
use_lora = st.sidebar.checkbox(
    "Use Fine-Tuned LoRA Adapter",
    value=True,
    help="Loads models/qwen3-1.7b-lora fine-tuned on verification, repair, and abstention.",
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

tab_qa, tab_eval, tab_finetune, tab_about = st.tabs(
    ["🩺 Q&A", "📊 Evaluation", "🔧 Fine-tuning", "ℹ️ About"]
)

# =============================================================================
# TAB 1 — Q&A
# =============================================================================
with tab_qa:
    st.title("Risk-Aware Medical Answer Verification")
    st.caption("Qwen3-1.7B · Evidence retrieval · Selective repair · Research prototype")

    question = st.text_area("Medical question", placeholder="What should someone know about …", height=100)

    if st.button("Generate and verify", type="primary", disabled=not question.strip()):
        pipe = get_pipeline(use_lora=use_lora)
        retriever = get_retriever() if use_hybrid else None

        with st.spinner("Generating, retrieving evidence, and checking claims …"):
            try:
                result = pipe.run(question, retriever=retriever)
                result["retriever"] = "hybrid_rrf" if retriever is not None else "bm25"

                # --- Final answer ---
                st.subheader("Final response")
                badge = "🔀 Hybrid RRF" if result.get("retriever") == "hybrid_rrf" else "🔡 BM25"
                st.caption(f"Retrieval: {badge}")
                st.write(result["answer"])

                if result["abstained"]:
                    st.warning("⚠️ Insufficient evidence for a safe answer; abstained from repair.")

                # --- Verification details ---
                st.subheader("Verification details")
                st.caption(result["note"])

                cols = st.columns([3, 1, 1])
                cols[0].markdown("**Claim**")
                cols[1].markdown("**Risk**")
                cols[2].markdown("**Status**")

                for i, claim in enumerate(result["claims"], 1):
                    risk_color = "🔴" if claim["risk"] == "high" else "🟡"
                    status_icon = "✅" if claim["status"] == "potentially_supported" else "❓"
                    with st.expander(f"Claim {i}: {claim['claim'][:80]}…"):
                        st.write(claim["claim"])
                        st.caption(
                            f"{risk_color} Risk: {claim['risk']} · "
                            f"{status_icon} {claim['status']} · "
                            f"NLI: {claim['verifier_label']} ({claim['confidence']:.0%}) · "
                            f"Source: {claim['evidence_source'] or 'none'}"
                        )

                # --- Retrieved evidence ---
                with st.expander(f"Retrieved evidence ({len(result['evidence'])} passages)"):
                    for item in result["evidence"]:
                        retriever_tag = item.get("retriever", "bm25")
                        corpus_tag = item.get("corpus", "MedQuAD")
                        st.markdown(
                            f"**{item['source']}** · score {item['score']:.4f} "
                            f"· `{retriever_tag}` · `{corpus_tag}`"
                        )
                        st.write(item["text"][:500])
                        st.divider()

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.info("Check README setup instructions and model/data paths.")

# =============================================================================
# TAB 2 — Evaluation
# =============================================================================
with tab_eval:
    st.header("📊 PubMedQA Evaluation")
    st.markdown(
        "Compare three configurations on PubMedQA labeled questions:\n"
        "1. **Baseline** — Qwen only\n"
        "2. **Retrieval** — Qwen + retrieval\n"
        "3. **Full** — Qwen + retrieval + NLI verification + repair/abstention"
    )

    results_path = Path(__file__).parent / "results" / "eval_results.json"

    col1, col2 = st.columns([2, 1])
    with col1:
        n_samples = st.slider("Evaluation samples", 5, 100, 20)
        configs = st.multiselect(
            "Configurations to run",
            ["baseline", "retrieval", "full"],
            default=["baseline", "retrieval", "full"],
        )
    with col2:
        run_eval = st.button("▶ Run evaluation", type="primary")

    if results_path.exists():
        with st.expander("📂 Load previously saved results"):
            try:
                saved = json.loads(results_path.read_text())
                st.json(saved)
                # Summary table
                rows = []
                for cfg, r in saved.items():
                    rows.append({
                        "Config": cfg,
                        "Accuracy": r.get("accuracy", "-"),
                        "Abstention": r.get("abstention_rate", "-"),
                        "Avg time (s)": r.get("avg_time_s", "-"),
                        "N": r.get("n", "-"),
                    })
                import pandas as pd
                st.dataframe(pd.DataFrame(rows).set_index("Config"))
            except Exception:
                pass

    if run_eval and configs:
        pipe = get_pipeline(use_lora=use_lora)
        retriever = get_retriever() if use_hybrid else None
        from evaluate import Evaluator

        evaluator = Evaluator(pipe, retriever=retriever, max_samples=n_samples)
        all_results = {}
        progress = st.progress(0.0, text="Running evaluation …")

        for i, cfg in enumerate(configs):
            st.info(f"Running: **{cfg}** …")
            if cfg == "baseline":
                r = evaluator.run_baseline()
            elif cfg == "retrieval":
                r = evaluator.run_retrieval()
            else:
                r = evaluator.run_full()
            all_results[cfg] = r
            progress.progress((i + 1) / len(configs), text=f"Done: {cfg}")

        st.success("Evaluation complete!")

        # Summary table
        import pandas as pd
        rows = [
            {
                "Config": cfg,
                "Accuracy": r.get("accuracy"),
                "Abstention": r.get("abstention_rate"),
                "Avg NLI score": r.get("avg_entailment_score", "-"),
                "Avg retrieval score": r.get("avg_retrieval_score", "-"),
                "Avg time (s)": r.get("avg_time_s"),
                "N": r.get("n"),
            }
            for cfg, r in all_results.items()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Config"))

        with st.expander("Full JSON results"):
            st.json(all_results)

        # Save
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        st.caption(f"Saved to {results_path}")

# =============================================================================
# TAB 3 — Fine-tuning
# =============================================================================
with tab_finetune:
    st.header("🔧 LoRA Fine-tuning")

    adapter_dir = Path(__file__).parent / "models" / "qwen3-1.7b-lora"
    finetune_data = Path(__file__).parent / "data" / "finetune_examples.json"

    st.markdown(
        "Fine-tune Qwen3-1.7B with LoRA/QLoRA for the three medical reasoning tasks:\n"
        "- **Claim verification** (entailment classification)\n"
        "- **Grounded repair** (evidence-grounded answer rewriting)\n"
        "- **Abstention** (high-risk claim detection)\n\n"
        "Run from a terminal with GPU access:"
    )

    st.code(
        "# Step 1 — validate data (CPU-safe dry run)\n"
        "python finetune.py --dry-run\n\n"
        "# Step 2 — full fine-tuning (requires GPU with ≥ 6 GB VRAM)\n"
        "python finetune.py --epochs 3 --batch-size 2 --lora-r 16\n\n"
        "# Step 3 — optional: merge adapter into full model\n"
        "python finetune.py --merge --adapter models/qwen3-1.7b-lora",
        language="bash",
    )

    # Training data editor
    st.subheader("Training data")
    if finetune_data.exists():
        try:
            examples = json.loads(finetune_data.read_text(encoding="utf-8"))
            st.caption(f"{len(examples)} examples in {finetune_data}")
            task_counts = {}
            for ex in examples:
                task_counts[ex.get("task", "unknown")] = task_counts.get(ex.get("task", "unknown"), 0) + 1
            st.bar_chart(task_counts)

            # Let user add a new example
            with st.expander("➕ Add a training example"):
                new_task = st.selectbox("Task", ["claim_verification", "grounded_repair", "abstention"])
                new_instruction = st.text_area("Instruction")
                new_input = st.text_area("Input")
                new_output = st.text_area("Output")
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
    else:
        st.info("No training data yet. Run `python finetune.py --dry-run` to generate the seed dataset.")
        if st.button("Generate seed dataset"):
            try:
                from finetune import _ensure_data
                _ensure_data()
                st.success(f"Seed data saved to {finetune_data}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # Adapter status
    st.subheader("Adapter status")
    if adapter_dir.exists() and any(adapter_dir.iterdir()):
        st.success(f"✅ LoRA adapter found at `{adapter_dir}`")
        adapter_files = list(adapter_dir.iterdir())
        st.caption(f"Files: {', '.join(f.name for f in adapter_files[:8])}")
    else:
        st.warning("⏳ No LoRA adapter found — fine-tuning has not been run yet.")

# =============================================================================
# TAB 4 — About
# =============================================================================
with tab_about:
    st.header("ℹ️ About this prototype")
    st.markdown("""
**Risk-Aware Medical Answer Verification** is a research prototype studying how a language model can:

1. Draft answers to medical information questions (Qwen3-1.7B)
2. Retrieve supporting evidence (BM25 lexical + optional PubMedBERT/FAISS hybrid)
3. Verify individual claims with an NLI model (`cross-encoder/nli-deberta-v3-small`)
4. Selectively repair low-risk unverified claims or abstain on high-risk ones

---

### Pipeline overview

```
User question
     │
     ▼
Qwen3-1.7B draft answer
     │
     ├──► Hybrid retrieval (BM25 + FAISS/PubMedBERT) ──► Evidence passages
     │
     ├──► Claim splitter → NLI verifier → risk classifier
     │
     └──► Repair engine / Abstention
                   │
                   ▼
            Final response
```

### Status
| Component | Status |
|---|---|
| Qwen3-1.7B generation | ✅ Implemented |
| MedQuAD BM25 retrieval | ✅ Implemented |
| NLI claim verification | ✅ Implemented |
| Repair / Abstention | ✅ Implemented |
| PubMedBERT dense retrieval (FAISS) | ✅ Implemented |
| PubMedQA integration + evaluation | ✅ Implemented |
| LoRA fine-tuning script | ✅ Implemented |
| LoRA adapter trained | ⏳ Not yet run (requires GPU session) |
| Three-way comparison | ✅ Implemented (run via Evaluation tab) |

> **Research demo only.** Not a medical device. Not clinically validated.
""")

    with st.expander("How to interpret labels"):
        st.markdown("""
- **potentially_supported** — NLI entailment score ≥ 0.70. Automated screening only; not proof of correctness.
- **unverified** — NLI could not confirm the claim against retrieved evidence.
- **neutral** — NLI verifier found no strong entailment or contradiction.
- **high risk** — Claim contains keywords related to medication dosage, diagnosis, or emergency care.
- **abstained** — Pipeline declined to answer due to high-risk unverifiable claims or absent evidence.
""")
