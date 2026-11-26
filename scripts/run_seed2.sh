#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf8
PY=venv/Scripts/python.exe
while [ ! -f /tmp/after.done ]; do sleep 10; done
$PY -u src/dump_fewshot.py > /tmp/fewshot.log 2>&1
$PY -u src/train_encoder_baseline.py > /tmp/deberta_ft.log 2>&1
echo DONE_FS > /tmp/fs.done
$PY -u src/train_claim_verifier.py --n-real 4000 --n-synth 2500 --epochs 2 --seed 43 --out models/claimverify-seed43 > /tmp/train_seed43.log 2>&1
$PY -u src/dump_predictions.py --adapter models/claimverify-seed43 --tag seed43 > /tmp/dump_seed43.log 2>&1
echo DONE3 > /tmp/seed2.done
