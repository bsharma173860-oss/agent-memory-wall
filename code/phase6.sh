#!/bin/bash
# Phase 6: production agent traffic (Mooncake/Kimi tool-agent trace), after phase5.sh. Edit only before it starts.
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
gpufree() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done; }
echo "##### E8 predictions (block LRU on the sampled trace)"
$PY mooncake_replay.py --model Qwen/Qwen3-32B --variants variants-mooncake.json --trace traces/toolagent_trace.jsonl \
    --fractions 0.25 --window 900 --predict-only
echo "##### E8a production tool-agent traffic, 1/4 of the trace, 15 minutes, five policies (H23-H25)"
gpufree; $PY mooncake_replay.py --model Qwen/Qwen3-32B --variants variants-mooncake.json --trace traces/toolagent_trace.jsonl \
    --fractions 0.25 --window 900 --drain 240 --rounds 1 --cpu --out mooncake-r1.json
echo "##### E8b round 2 (reversed order) for GPU-only, no cache, 160 GB"
gpufree; $PY mooncake_replay.py --model Qwen/Qwen3-32B --variants variants-mooncake-r2.json --trace traces/toolagent_trace.jsonl \
    --fractions 0.25 --window 900 --drain 240 --round-start 2 --rounds 2 --cpu --out mooncake-r2.json
python3 -m zipfile -c phase6-results.zip mooncake-r*.json *-mc-r*-server.log phase6.log 2>/dev/null
cp phase6-results.zip / ; echo "##### PHASE6 DONE: phase6-results.zip (also in /)"
