#!/bin/bash
# Phase 4, after phase3.sh. Edit only before it starts.
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
echo "##### E5-0b rerun of the full stack (160 GB + formula) round 2: it hit CUDA OOM because the previous engine had not exited"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done
$PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-fullstack-rerun.json --sessions 48,96 --pauses 5 \
    --duration 150 --rounds 1 --cpu --out agents-fullstack-rerun.json
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done
echo "##### E6 BFCL multi-turn (base + long context): correct answers per kWh, GPU cache vs no cache (H12)"
$PY bfcl_energy.py --variants variants-bfcl.json --categories multi_turn_base,multi_turn_long_context \
    --threads 100 --rounds 2 --out bfcl.json
python3 -m zipfile -c phase4-results.zip agents-fullstack-rerun* bfcl.json bfcl-scores *-bfcl-r*-server.log phase4.log 2>/dev/null
cp phase4-results.zip / ; echo "##### PHASE4 DONE: phase4-results.zip (also in /)"
