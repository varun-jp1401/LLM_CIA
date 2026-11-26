#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf8
PY=venv/Scripts/python.exe
$PY -u src/dump_predictions.py --adapter models/qwen3-1.7b-lora-claimverify --tag mixed > /tmp/dump_mixed.log 2>&1
$PY -u src/train_claim_verifier.py --n-real 4000 --n-synth 0 --epochs 2 --out models/ablation-real-only > /tmp/train_real.log 2>&1
$PY -u src/dump_predictions.py --adapter models/ablation-real-only --tag real_only > /tmp/dump_real.log 2>&1
$PY -u src/train_claim_verifier.py --n-real 0 --n-synth 2500 --epochs 2 --out models/ablation-synth-only > /tmp/train_synth.log 2>&1
$PY -u src/dump_predictions.py --adapter models/ablation-synth-only --tag synth_only > /tmp/dump_synth.log 2>&1
echo ALLDONE > /tmp/ablations.done
