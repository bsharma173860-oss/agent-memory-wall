#!/bin/bash
# Phase 7 (filler so the GPU never idles; may be replaced before it starts): production traffic, second window.
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
gpufree() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done; }
echo "##### E9 production tool-agent traffic, second window (minutes 30-45), 1/4 and 1/8 of the trace"
$PY mooncake_replay.py --model Qwen/Qwen3-32B --variants variants-mooncake-r2.json --trace traces/toolagent_trace.jsonl \
    --fractions 0.125,0.25 --offset 1800 --window 900 --predict-only
gpufree; $PY mooncake_replay.py --model Qwen/Qwen3-32B --variants variants-mooncake-r2.json --trace traces/toolagent_trace.jsonl \
    --fractions 0.125,0.25 --offset 1800 --window 900 --drain 240 --rounds 1 --cpu --out mooncake-w2.json
echo "##### R7b rerun: vLLM formula on Qwen3-8B, round 1 (its engine OOMed behind the previous one)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-8B --variants variants-p5-8b-formula.json --sessions 64,128,256 --pauses 5 \
    --duration 150 --rounds 1 --cpu --out agents-p5-8b-formula-r1.json
echo "##### R5b BFCL rerun (H12): result dir now created, GPU-free wait between engines"
gpufree; $PY bfcl_energy.py --variants variants-bfcl.json --categories multi_turn_base,multi_turn_long_context \
    --threads 100 --rounds 2 --out bfcl.json
python3 -m zipfile -c phase7-results.zip mooncake-w2.json agents-p5-8b-formula-r1* bfcl.json bfcl-scores *-bfcl-r*-server.log phase7.log 2>/dev/null
cp phase7-results.zip / ; echo "##### PHASE7 DONE: phase7-results.zip (also in /)"
