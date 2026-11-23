#!/bin/bash
# After run_all.sh: (1) vLLM offload at 70 GiB (/dev/shm is 75 GiB), (2) session admission, 16 slots,
# (3) SGLang eviction policies LFU and SLRU. Results zipped to /.
set -uo pipefail
cd /workspace; PY=/workspace/venv-bench/bin/python
sed 's#/workspace/agents.json#/workspace/agents-offload.json#' cpu_log2.py > cpu_log3.py
python3 cpu_log3.py cpu-time-offload.csv & LOG=$!
echo "##### E3 offload rerun (70 GiB)"
$PY agent_energy.py --model Qwen/Qwen3-32B --variants variants-offload.json --sessions 16,48,96 --pauses 5,30 \
    --duration 150 --rounds 2 --cpu --out agents-offload.json
echo "##### E3b session admission"
$PY agent_admit.py --model Qwen/Qwen3-32B --variants variants-admit.json --sessions 48,96 --pauses 5,30 \
    --duration 150 --rounds 2 --cpu --out agents-admit.json
echo "##### E3c SGLang eviction policies"
$PY agent_energy.py --model Qwen/Qwen3-32B --variants variants-evict.json --sessions 16,48,96 --pauses 5,30 \
    --duration 150 --rounds 2 --cpu --out agents-evict.json
echo "##### E3d cliff position (vLLM GPU-only, 24/32/40 sessions)"
python3 -c "import json;json.dump([v for v in json.load(open('variants-agent.json')) if v['name']=='vllm-gpu'],open('variants-cliff.json','w'))"
$PY agent_energy.py --model Qwen/Qwen3-32B --variants variants-cliff.json --sessions 24,32,40 --pauses 5,30 \
    --duration 150 --rounds 2 --cpu --out agents-cliff.json
echo "##### E3e SGLang offload with a host tier sized for 48 sessions (160 GB)"
python3 -c "import json;json.dump([{'name':'sglang-offload160','engine':'sglang','policy':'offload','python':'/workspace/venv-sglang/bin/python','flags':['--enable-hierarchical-cache','--hicache-size','160']}],open('variants-host160.json','w'))"
$PY agent_energy.py --model Qwen/Qwen3-32B --variants variants-host160.json --sessions 16,48,96 --pauses 5,30 \
    --duration 150 --rounds 2 --cpu --out agents-host160.json
echo "##### E3f transfer or recompute: J per token of GPU hit, host restore, recompute"
$PY transfer_energy.py --model Qwen/Qwen3-32B --variants variants-transfer.json --lengths 4000,8000,16000 --k 8 \
    --flush-tokens 300000 --reps 3 --cpu --out transfer.json
echo "##### E3g new random sessions (seeds s1, s2): vLLM GPU cache vs no cache"
python3 -c "import json;json.dump([v for v in json.load(open('variants-agent.json')) if v['name'] in ('vllm-gpu','vllm-nocache')],open('variants-seed.json','w'))"
for SEED in s1 s2; do
  $PY agent_seed.py --model Qwen/Qwen3-32B --variants variants-seed.json --sessions 16,48 --pauses 5,30 \
      --duration 150 --rounds 1 --cpu --seed $SEED --out agents-seed-$SEED.json
done
kill $LOG
python3 -m zipfile -c extra-results.zip agents-offload* agents-admit* agents-evict* agents-cliff* agents-host160* transfer* agents-seed* cpu-time*.csv *-agent-r*-server.log extra.log 2>/dev/null
cp extra-results.zip / ; echo "##### EXTRA DONE: extra-results.zip (also in /)"
