#!/bin/bash
# Everything, most important first, so a stop part-way still leaves the main results:
#   E2  re-prefill cost by prompt length x load (both engines; ~1.5 h)
#   E3  agent sessions: GPU-only vs offload (80 GB host) vs no cache (both engines; ~4.5 h)   <- the core of the paper
#   P2  part 2: default vs tuned, open-loop chat and RAG (both engines; ~2.5 h)
# Each stage saves after every step; a failed stage does not stop the next. Results are copied to / at the end.
set -uo pipefail
W=${W:-/workspace}; MODEL=${MODEL:-Qwen/Qwen3-32B}; PRICE=${PRICE:-0}; PY=$W/venv-bench/bin/python
cd "$W"
E="--engine vllm=$W/venv-vllm/bin/python --engine sglang=$W/venv-sglang/bin/python"
for L in 1600 12600; do   # ~2k and ~16k prompt tokens (the 8k-token point is part 1)
  echo "##### E2: prompt ~$L target" | tee -a run_all.log
  $PY engine_energy.py --model "$MODEL" $E --workloads long_miss,long_hit --long-tokens $L --max-len 20480 \
      --users 1,4,16,64 --per-user 4 --price "$PRICE" --out e2-len$L.json 2>&1 | tee -a run_all.log
done
echo "##### E3: agents" | tee -a run_all.log
$PY agent_energy.py --model "$MODEL" --variants variants-agent.json --sessions 16,48,96 --pauses 5,30 --duration 150 \
    --rounds 2 --cpu --price "$PRICE" --out agents.json 2>&1 | tee -a run_all.log
echo "##### P2: part 2" | tee -a run_all.log
$PY engine_energy2.py --model "$MODEL" --variants variants.json --rounds 2 --cpu --price "$PRICE" --out part2.json 2>&1 | tee -a run_all.log
python3 -m zipfile -c all-results.zip e2-len*.json agents*.json agents.csv part2*.json part2.csv run_all.log *server.log 2>/dev/null
cp all-results.zip / 2>/dev/null
echo "##### ALL DONE: all-results.zip (also in /)" | tee -a run_all.log
