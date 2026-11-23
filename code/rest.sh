#!/bin/bash
# Single controller for everything left (7 Nov 08:45 UTC), in order of value. Replaces the phase 5-7 chains:
# two copies of phase5.sh ran at once from 07:11 (an old watcher plus the new chain), so every phase 5 result
# before 08:40 is void (moved to void-p5/). BFCL failed because phase4.sh had no GPU-free wait before it.
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
gpufree() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done; sleep 5; }
echo "##### R1 phase 6: production tool-agent traffic (H23-H25)"
gpufree; bash phase6.sh > phase6.log 2>&1
echo "##### R2 E7c real trajectories at 192/288: full stack, 80 GB write_back, vLLM prefix-aware formula (H21)"
gpufree; $PY agent_replay.py --model Qwen/Qwen3-32B --variants variants-p5-real.json --traces traj/swe-traj-400.jsonl \
    --sessions 192,288 --pauses 5 --duration 150 --rounds 2 --cpu --out agents-p5-real.json
echo "##### R3 E7a full stack with rotation over 10 minutes, 96 sessions (H19)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-p5-fair.json --sessions 96 --pauses 5 \
    --duration 600 --rounds 2 --cpu --out agents-p5-fair.json
echo "##### R4 E7b cheap full stack: 80 GB write_back + formula, 48/96 sessions (H20)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-32B --variants variants-p5-cheap.json --sessions 48,96 --pauses 5 \
    --duration 150 --rounds 2 --cpu --out agents-p5-cheap.json
echo "##### R5 E6 BFCL multi-turn: correct answers per kWh, GPU cache vs no cache (H12)"
gpufree; $PY bfcl_energy.py --variants variants-bfcl.json --categories multi_turn_base,multi_turn_long_context \
    --threads 100 --rounds 2 --out bfcl.json
echo "##### R6 E7d third round for headline cells"
gpufree; $PY agent_energy.py --model Qwen/Qwen3-32B --variants variants-p5-r3.json --sessions 16,48,96 --pauses 5 \
    --duration 150 --rounds 1 --cpu --out agents-p5-r3.json
echo "##### R7 E7f Qwen3-8B: cliff and fixes at 64/128/256 sessions (H22)"
gpufree; $PY agent_ctrl.py --model Qwen/Qwen3-8B --variants variants-p5-8b.json --sessions 64,128,256 --pauses 5 \
    --duration 150 --rounds 2 --cpu --out agents-p5-8b.json
python3 -m zipfile -c rest-results.zip agents-p5-* bfcl.json bfcl-scores *-agent-r*-server.log *-bfcl-r*-server.log rest.log 2>/dev/null
cp rest-results.zip / ; echo "##### REST DONE (R2-R7): rest-results.zip"
echo "##### R8 phase 7 (filler): production trace, second window"
gpufree; bash phase7.sh > phase7.log 2>&1
echo "##### ALL DONE"
