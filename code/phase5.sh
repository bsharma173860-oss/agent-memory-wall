#!/bin/bash
# Phase 5 (15 extra GPU hours), after phase4.sh. Edit only before it starts.
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
gpufree() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done; }
echo "##### E7a full stack with rotation over 10 minutes, 96 sessions (H19)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-p5-fair.json --sessions 96 --pauses 5 \
    --duration 600 --rounds 2 --cpu --out agents-p5-fair.json
echo "##### E7b cheap full stack: 80 GB write_back + formula, 48/96 sessions (H20)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-p5-cheap.json --sessions 48,96 --pauses 5 \
    --duration 150 --rounds 2 --cpu --out agents-p5-cheap.json
echo "##### E7c real trajectories at 192/288: full stack and 80 GB write_back (H21)"
gpufree; $PY agent_replay.py --model Qwen/Qwen3-32B --variants variants-p5-real.json --traces traj/swe-traj-400.jsonl \
    --sessions 192,288 --pauses 5 --duration 150 --rounds 2 --cpu --out agents-p5-real.json
echo "##### E7d third round for headline cells (cliff, cache vs no cache, 160 GB)"
gpufree; $PY agent_energy.py --model Qwen/Qwen3-32B --variants variants-p5-r3.json --sessions 16,48,96 --pauses 5 \
    --duration 150 --rounds 1 --cpu --out agents-p5-r3.json
echo "##### E7e second dense model: Qwen3-8B download"
/workspace/venv-vllm/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))"
echo "##### E7f Qwen3-8B: cliff and fixes at 64/128/256 sessions (H22)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-8B --variants variants-p5-8b.json --sessions 64,128,256 --pauses 5 \
    --duration 150 --rounds 2 --cpu --out agents-p5-8b.json
python3 -m zipfile -c phase5-results.zip agents-p5-* *-agent-r*-server.log phase5.log 2>/dev/null
cp phase5-results.zip / ; echo "##### PHASE5 DONE: phase5-results.zip (also in /)"
