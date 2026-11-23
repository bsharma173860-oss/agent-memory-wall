#!/bin/bash
# Run after setup.sh, before run_all.sh (about a minute, no model is loaded): checks that every flag the variants use
# exists in the installed vLLM and SGLang, and prints each engine's offload-related options. Send the output if a
# variant shows MISSING.
cd "${W:-/workspace}"
venv-bench/bin/python - <<'PY'
import json, sys
sys.path.insert(0, ".")
from engine_energy2 import check_flags
for f in ("variants.json", "variants-agent.json"):
    for v in json.load(open(f)):
        missing = check_flags(v)
        print(f"{f:<22} {v['name']:<16} {'OK' if not missing else 'MISSING ' + str(missing)}")
PY
for e in vllm sglang; do
  m=$([ $e = vllm ] && echo vllm.entrypoints.openai.api_server || echo sglang.launch_server)
  echo "--- $e options about offload / host cache / prefix cache:"
  venv-$e/bin/python -m $m --help 2>&1 | grep -i -E -- "--[a-z-]*(offload|hicache|hierarchical|prefix-caching|radix)" | sed 's/^ */  /' | head -20
done
