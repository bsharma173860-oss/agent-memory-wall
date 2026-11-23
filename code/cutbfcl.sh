#!/bin/bash
# Budget guard (7 Nov 20:35 UTC): stop BFCL after its round-1 pair (GPU cache + no cache) so phase 8 fits the balance.
cd /workspace
until [ "$(grep -c 'evaluate rc' phase7.log)" -ge 2 ]; do sleep 20; done
echo "$(date -u) round-1 pair done; stopping BFCL round 2" >> cutbfcl.log
for p in $(pgrep -f bfcl_energy.py) $(pgrep -f "venv-bfcl/bin/bfcl"); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f "vllm.entrypoints.openai.api_server"); do kill "$p" 2>/dev/null; done
sleep 20
for p in $(pgrep -f "vllm.entrypoints.openai.api_server"); do kill -9 "$p" 2>/dev/null; done
echo "$(date -u) done" >> cutbfcl.log
