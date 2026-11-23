#!/bin/bash
# One H200 box: three separate Python environments (vLLM, SGLang and the benchmark with joules never share
# packages), then the model download. About 15-25 minutes, mostly downloads.
set -euo pipefail
W=${W:-/workspace}; MODEL=${MODEL:-Qwen/Qwen3-32B}
cd "$W"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv
df -h "$W" | tail -1
uv venv -q -p 3.12 venv-bench  && uv pip install -q -p venv-bench/bin/python landauer-gap huggingface_hub
uv venv -q -p 3.12 venv-vllm   && uv pip install -q -p venv-vllm/bin/python vllm
uv venv -q -p 3.12 venv-sglang && (uv pip install -q -p venv-sglang/bin/python "sglang[all]" || uv pip install -q -p venv-sglang/bin/python sglang)
venv-bench/bin/python -c "import joules; print('joules', joules.__version__)"
venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
venv-sglang/bin/python -c "import sglang; print('sglang', sglang.__version__)"
venv-bench/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken']))"
echo "SETUP DONE"
