#!/bin/bash
# Phase 9 (approved 8 Nov 2026, budget $8): the adaptive gate in 10-minute steps (H28). Hard stop after 65 minutes. Scope cut to 8B/256 and 32B-long/32 (budget, after the first box's driver was too old).
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
gpufree() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done; sleep 5; }
DEADLINE=$(( $(date +%s) + 3900 ))
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "##### E12a Qwen3-8B, GPU cache vs adaptive gate, 256 sessions, 600 s steps (H28)"
gpufree; timeout $(left) $PY agent_ctrl.py --model Qwen/Qwen3-8B --variants variants-p9-8b.json --sessions 256 --pauses 5 \
    --duration 600 --rounds 1 --cpu --out agents-p9-8b.json
echo "##### E12b Qwen3-32B long context, GPU cache vs adaptive gate, 32 sessions, 600 s steps (H28)"
[ "$(left)" -gt 1500 ] && { gpufree; timeout $(left) $PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-p9-lc.json --sessions 32 --pauses 5 \
    --duration 600 --context 20000 --max-len 40960 --rounds 1 --cpu --out agents-p9-lc.json; } || echo "skipped E12b: not enough time left"
python3 -m zipfile -c phase9-results.zip agents-p9-* *-10m-agent-r*-server.log phase9.log 2>/dev/null
cp phase9-results.zip / ; echo "##### PHASE9 DONE: phase9-results.zip (also in /). GPU queue empty: stop the instance."
