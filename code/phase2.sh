#!/bin/bash
# Phase 2, after extra.sh. Edit only before it starts (bash reads a running script as it goes).
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
echo "##### E4a controller: formula (K = C/s, live) vs CONCUR-style AIMD vs none (vLLM)"
$PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-ctrl.json --sessions 48,96 --pauses 5,30 \
    --duration 150 --rounds 2 --cpu --out agents-ctrl.json
echo "##### E4b real agent trajectories (SWE-agent, 400 recorded runs): 7 policies, 16/48/96 sessions"
$PY agent_replay.py --model Qwen/Qwen3-32B --variants variants-replay.json --traces traj/swe-traj-400.jsonl \
    --sessions 16,48,96 --pauses 5 --duration 150 --rounds 2 --cpu --out agents-real.json
echo "##### E4c real trajectories: SGLang LFU and SLRU eviction, 16/48/96 sessions"
$PY agent_replay.py --model Qwen/Qwen3-32B --variants variants-replay2.json --traces traj/swe-traj-400.jsonl \
    --sessions 16,48,96 --pauses 5 --duration 150 --rounds 2 --cpu --out agents-real-evict.json
echo "##### E4d real trajectories: where the cliff is (vLLM GPU-only, 24/32/40 sessions)"
python3 -c "import json;json.dump([v for v in json.load(open('variants-replay.json')) if v['name']=='vllm-gpu-real'],open('variants-replay-cliff.json','w'))"
$PY agent_replay.py --model Qwen/Qwen3-32B --variants variants-replay-cliff.json --traces traj/swe-traj-400.jsonl \
    --sessions 24,32,40 --pauses 5 --duration 150 --rounds 2 --cpu --out agents-real-cliff.json
echo "##### E4e does the cache expire with time? (no pressure, resend after 0/60/300/900/1500 s)"
$PY retention_test.py --model Qwen/Qwen3-32B --variants variants-retention.json --delays 0,60,300,900,1500 --out retention.json
echo "##### E4f fairness over 10 minutes: GPU-only vs static gate vs formula (48 sessions, pause 5 s)"
$PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-long.json --sessions 48 --pauses 5 \
    --duration 600 --rounds 2 --cpu --out agents-long.json
python3 -m zipfile -c phase2-results.zip agents-ctrl* agents-long* agents-real* retention* traj/swe-traj-400.jsonl *-agent-r*-server.log phase2.log 2>/dev/null
cp phase2-results.zip / ; echo "##### PHASE2 DONE: phase2-results.zip (also in /)"
