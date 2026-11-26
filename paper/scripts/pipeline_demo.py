"""Illustrative walk-through of the deployed pipeline (pipeline.py) on a few questions. Not an evaluation."""
import json
from pathlib import Path
from pipeline import MedicalPipeline

QUESTIONS = [
    "What are the symptoms of glaucoma?",
    "What causes iron deficiency anemia?",
    "How is high blood pressure treated?",
    "Should I double my warfarin dose if I missed yesterday's dose?",
]
pl = MedicalPipeline(adapter_path=None)
out = []
for q in QUESTIONS:
    r = pl.run(q)
    out.append({"question": q, "abstained": r["abstained"], "n_evidence": len(r["evidence"]), "draft": r["draft"], "answer": r["answer"],
                "claims": [{k: c[k] for k in ("claim", "verifier_label", "confidence", "risk", "status")} for c in r["claims"]]})
    print(q, "| claims", len(r["claims"]), "| abstained", r["abstained"], flush=True)
Path("results/pipeline_demo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
