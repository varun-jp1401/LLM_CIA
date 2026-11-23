"""Risk-aware medical answer verification and selective repair demo."""
from __future__ import annotations

import streamlit as st

from pipeline import MedicalPipeline


@st.cache_resource(show_spinner="Loading Qwen3-1.7B…")
def get_pipeline():
    return MedicalPipeline()


st.set_page_config(page_title="Medical Answer Verification", page_icon="🩺", layout="wide")
st.title("Risk-Aware Medical Answer Verification")
st.caption("Research prototype · Qwen3-1.7B · Evidence retrieval · Selective repair")
question = st.text_area("Medical question", placeholder="What should someone know about…", height=100)
if st.button("Generate and verify", type="primary", disabled=not question.strip()):
    with st.spinner("Generating, retrieving evidence, and checking claims…"):
        try:
            result = get_pipeline().run(question)
            st.subheader("Final response")
            st.write(result["answer"])
            if result["abstained"]:
                st.warning("Insufficient evidence for a safe answer; abstained from repair.")
            st.subheader("Verification details")
            st.caption(result["note"])
            for i, claim in enumerate(result["claims"], 1):
                st.markdown(f"**Claim {i} · {claim['risk']} risk · {claim['status']}**  \n{claim['claim']}")
                st.caption(f"NLI: {claim['verifier_label']} ({claim['confidence']:.0%}) · Source: {claim['evidence_source'] or 'none'}")
            with st.expander(f"Retrieved evidence ({len(result['evidence'])})"):
                for item in result["evidence"]:
                    st.markdown(f"**{item['source']}** · relevance {item['score']:.3f}")
                    st.write(item["text"])
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.info("Check README setup instructions and model/data paths.")

with st.expander("How this demo works"):
    st.markdown("1. Qwen drafts an answer. 2. The answer is split into claims. 3. MedQuAD evidence is retrieved. 4. Claims are checked for textual support and risk. 5. Qwen selectively repairs flagged claims or abstains when evidence is weak.")
