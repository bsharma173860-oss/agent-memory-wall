# Settings of the paper's gate runs, and sizes for other GPUs

## As run in the paper (the 10-minute steps, `code/phase9.sh`)

| | Qwen3-8B | Qwen3-32B, long context |
|---|---|---|
| GPU | 1x NVIDIA H200 NVL (143,771 MiB, 600 W limit, driver 595.99.02) | same |
| Engine | vLLM 0.33.0, Python 3.12 | same |
| Meter | joules 0.8.2 (`landauer-gap==0.8.2`), NVML energy counter, GPU board energy | same |
| Engine command | `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --served-model-name bench --port 18000 --max-model-len 32768 --gpu-memory-utilization 0.90 --enable-prefix-caching --enable-prompt-tokens-details` | the same with `--model Qwen/Qwen3-32B --max-model-len 40960` |
| KV capacity the engine reported | 777,872 tokens | 244,928 tokens |
| Sessions | 256 | 32 |
| Each session's start | ~8,000 tokens of synthetic repository text (first prompt about 10.6k tokens with the first tool result) | ~20,000 tokens (first prompt about 26k) |
| Each turn | 400 tokens of synthetic tool output appended, 200 output tokens, then a fixed one-line summary appended | same |
| Pauses between turns | log-normal, median 5 s, sigma 0.8, at most 6x the median | same |
| Step | 600 s per policy, one round, sessions started up to 0.25 s apart | same |
| Before each step | 4 warm-up requests, card temperature settled (60 s window), 30 s idle measured | same |
| Gate (`code/variants-p9-8b.json`, `code/variants-p9-lc.json`) | adaptive: theta 1.2, safety 0.9, 4 turns per admission, 600 tokens growth, 2 s interval; session size = 90th percentile of the last 64 prompts | the same, session size = their mean |

The baseline is the same command with no gate: the engine's own prefix cache in GPU memory.

`run_kit.py` sets every one of these as in the paper except two:
- the session count, which it takes from the table below;
- the GPU index and port, which you choose.

For SGLang the paper's command is the following. The paper ran its gate on vLLM only.

```
python -m sglang.launch_server --model-path MODEL --served-model-name bench --port 18000 --context-length MAX_LEN \
  --mem-fraction-static 0.88 --enable-metrics --enable-cache-report
```

## Sizes for H100 80 GB and MI300X 192 GB

These are sizes only, not results. The gate matters only when the working set is past the cache, so the session count
is scaled to keep the load against your card's KV cache about where it was in the paper.

`sizes.py` does this in four steps:
1. It takes the memory the engine kept for everything but the KV cache on the paper's H200, at
   `--gpu-memory-utilization 0.90`.
2. It assumes the same on your card.
3. From that, it estimates the KV room left on your card.
4. It scales the paper's session count by that KV room.

| Workload | GPU | KV room (tokens, estimate) | Sessions |
|---|---|---|---|
| 8b | H200 (the paper) | 777,872 | 256 |
| 8b | H100 80 GB | 379,715 | 125 |
| 8b | MI300X 192 GB | 1,116,028 | 367 |
| 32b-long | H200 (the paper) | 244,928 | 32 |
| 32b-long | H100 80 GB | 20,964 | does not fit: one full-length session needs more KV room than the card has left |
| 32b-long | MI300X 192 GB | 435,141 | 57 |

For another card, run `python3 -I gate-kit/sizes.py --gib <memory in GiB>`.

The engine's own figure is the one to trust: "GPU KV cache size" in its log, which also appears as
`kv_capacity_tokens` in the results file. If it differs much from the estimate, run again with
`--sessions` scaled by the ratio. The results file also records the peak working set over the cache. In the paper's
runs the gate was on for most of the step; if yours shows the gate seldom on, the load was below the cliff and the two
runs should look alike.

What changes on an MI300X:
- the engine is the ROCm build of vLLM;
- the energy meter is amdgpu's (see the README);
- nothing else.

The Qwen3-32B long-context workload does not fit on one H100 80 GB. Two H100s with tensor parallelism would hold it,
but the paper did not test that, and this kit does not size it.
