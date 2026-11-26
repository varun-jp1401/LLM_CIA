"""Constants and loaders shared by analysis.py, analysis_extra.py and analysis_calib.py."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
L = ["entailment", "neutral", "contradiction"]
SETS = ["medquad_test_iid", "medquad_test_ood", "healthver_test"]


def load(tag):
    """Per-example predictions dumped by dump_predictions.py & co. (None if the run does not exist)."""
    p = R / f"predictions_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
