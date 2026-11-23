#!/bin/bash
# Phase 9 on a fresh H200 NVL box: only what phase 9 needs, pinned to the versions of phases 1-8
# (vLLM 0.33.0, landauer-gap 0.8.2 = joules 0.8.2, Python 3.12), then Qwen3-8B and Qwen3-32B. About 10-15 minutes.
set -euo pipefail
W=${W:-/workspace}; cd "$W"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv
df -h "$W" | tail -1; free -g | head -2
uv venv -q -p 3.12 venv-bench && uv pip install -q -p venv-bench/bin/python "landauer-gap==0.8.2" huggingface_hub
uv venv -q -p 3.12 venv-vllm  && uv pip install -q -p venv-vllm/bin/python "vllm==0.33.0"
venv-bench/bin/python -c "import joules; print('joules', joules.__version__)"
venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
venv-vllm/bin/python -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda ok:', ok); assert ok, 'driver too old for this build'"
for M in Qwen/Qwen3-8B Qwen/Qwen3-32B; do
  venv-bench/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$M', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken']))"
done
echo "SETUP9 DONE"
