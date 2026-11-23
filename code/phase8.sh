#!/bin/bash
# Phase 8 (approved 7 Nov ~18:50 UTC): the memory wall at long context (H26) and brain v0, the adaptive controller (H27).
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
gpufree() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done; sleep 5; }
echo "##### E10a long context (~20k-token sessions), 4/8/16/32 sessions: GPU, no cache, formula, adaptive (H26, H27)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-longctx.json --sessions 4,8,16,32 --pauses 5 \
    --duration 150 --context 20000 --max-len 40960 --rounds 1 --cpu --out agents-longctx-r1.json
echo "##### E10b brain v0 on Qwen3-8B, 64/128/256 sessions, 2 rounds (H27)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-8B --variants variants-8b-adaptive.json --sessions 64,128,256 --pauses 5 \
    --duration 150 --rounds 2 --cpu --out agents-8b-adaptive.json
echo "##### E10c long context, round 2"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-longctx.json --sessions 4,8,16,32 --pauses 5 \
    --duration 150 --context 20000 --max-len 40960 --rounds 1 --cpu --out agents-longctx-r2.json
python3 -m zipfile -c phase8-results.zip agents-longctx-* agents-8b-adaptive* *-lc-agent-r*-server.log *adaptive-8b-agent-r*-server.log phase8.log 2>/dev/null
cp phase8-results.zip / ; echo "##### PHASE8 DONE: phase8-results.zip (also in /). GPU queue empty: stop the instance."
