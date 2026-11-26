#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf8
PY=venv/Scripts/python.exe
while [ ! -f /tmp/ablations.done ]; do sleep 10; done
$PY -u src/dump_predictions.py --adapter none --tag base > /tmp/dump_base.log 2>&1
$PY -u src/dump_predictions.py --adapter models/qwen3-1.7b-lora-claimverify --tag mixed --split val > /tmp/dump_mixed_val.log 2>&1
echo DONE2 > /tmp/after.done
