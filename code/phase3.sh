#!/bin/bash
# Phase 3, after phase2.sh. Edit only before it starts.
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
echo "##### E5-0 RAM sizing curve (SGLang host tier 120/240 GB) and full stack (160 GB + formula controller), 48/96 sessions"
$PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-sizing.json --sessions 48,96 --pauses 5 \
    --duration 150 --rounds 2 --cpu --out agents-sizing.json
echo "##### E5-1 real trajectories at high load (192/288 sessions): GPU, no cache, formula, SGLang 160 GB (H14b)"
$PY agent_replay.py --model Qwen/Qwen3-32B --variants variants-replay-hi.json --traces traj/swe-traj-400.jsonl \
    --sessions 192,288 --pauses 5 --duration 150 --rounds 2 --cpu --out agents-real-hi.json
echo "##### E5a idle GPU power, empty vs full KV cache (H11)"
$PY idle_cache.py --model Qwen/Qwen3-32B --variants variants-idle.json --capacity 244928 --out idle-cache.json
echo "##### E5b SGLang flag ablation: RAG at 6 and 8 req/s (exploratory)"
$PY engine_energy2.py --model Qwen/Qwen3-32B --variants variants-ablation.json --workloads rag --rag-rates 6,8 \
    --rounds 2 --cpu --out ablation.json
echo "##### E5c MoE model download (Qwen3-30B-A3B)"
/workspace/venv-vllm/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-30B-A3B'))"
echo "##### E5d MoE, synthetic sessions: GPU cache vs no cache, 32/96/192 sessions (H16)"
$PY agent_energy.py --model Qwen/Qwen3-30B-A3B --variants variants-moe.json --sessions 32,96,192 --pauses 5 \
    --duration 150 --rounds 2 --cpu --out agents-moe.json
echo "##### E5e MoE, real trajectories: GPU cache, no cache, formula controller (H16)"
$PY agent_replay.py --model Qwen/Qwen3-30B-A3B --variants variants-moe-real.json --traces traj/swe-traj-400.jsonl \
    --sessions 32,96,192 --pauses 5 --duration 150 --rounds 2 --cpu --out agents-moe-real.json
python3 -m zipfile -c phase3-results.zip agents-sizing* agents-real-hi* traj/swe-traj-400.jsonl idle-cache* ablation* agents-moe* *-idle-server.log phase3.log 2>/dev/null
cp phase3-results.zip / ; echo "##### PHASE3 DONE: phase3-results.zip (also in /)"
