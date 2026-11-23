# Gate kit: run the paper's admission gate on your own GPU

This folder lets you run the admission gate from *The Agent Memory Wall* ([paper](../paper/agent-memory-wall.pdf),
[doi:10.5281/zenodo.24146466](https://doi.org/10.5281/zenodo.24146466)) on your own vLLM or SGLang, against the same
engine with no gate, and send us the numbers. It uses the paper's own benchmark and gate (`../code/agent_ctrl.py`)
unchanged; this folder only sets it up, sizes it for your GPU and writes a small results file.

## What the gate does

AI agents pause while their tools run. While they pause, the engine keeps each agent's memory of the conversation
(its KV cache) only if nothing else needs the room. When the open sessions' memory together (their working set) grows
past what the GPU can cache, the cache stops working: in the paper, cache hits fell to half at about 1.2 times the
cache and were almost gone by about 1.55 times (with 5 s pauses), and every turn then rebuilt its context.

The gate is an admission rule in front of the engine. The engine is not changed.

- **While the working set fits under 1.2 times the cache, it does nothing:** every session gets in at once.
- **Once the working set is over 1.2 times the cache, it switches on:** only K sessions may send turns, with
  K = 0.9 × cache size / session size. The session size is the recent prompt size, plus 600 tokens of growth.
- **Rotation:** a session keeps its place through its pauses for 4 turns, so its memory stays warm. Then it gives the
  place up and queues again, first in, first out.
- **Timing:** the gate checks every 2 seconds.

## The cost, up front

**The gate trades waiting for throughput.** New sessions queue for a place.

In the paper's two 10-minute runs on one H200 (Qwen3-8B with 256 sessions; Qwen3-32B with 32 long sessions):
- the 90th percentile of **first-turn latency** went from **68 to 183 s** and from **131 to 253 s**;
- for resumed turns, including the wait to get back in, it went from 102 to 226 s and from 130 to 251 s.

In exchange, in those same two runs:
- every session was served;
- the gate completed 48% and 62% more turns;
- it used 42% and 50% less GPU energy per turn.

If your users cannot wait minutes for a first answer, the gate in this form is not for you.

What the paper did **not** test:
- pauses longer than 5 s (with 30 s pauses, hits last until about 2.2 times the cache, so the 1.2 trigger may be early);
- recorded or production traffic;
- SGLang with this gate;
- any GPU but the H200.

That last point is the reason for this kit.

## Run it

You need:
- one GPU, with nothing else running on it;
- the engine in its own Python environment;
- a second environment for the benchmark;
- about 30 minutes: two 10-minute runs, plus engine start-up and settling.

```bash
git clone https://github.com/bsharma173860-oss/agent-memory-wall && cd agent-memory-wall
uv venv -p 3.12 venv-vllm  && uv pip install -p venv-vllm/bin/python "vllm==0.33.0"            # the paper's engine
uv venv -p 3.12 venv-bench && uv pip install -p venv-bench/bin/python "landauer-gap==0.8.2"   # joules, the meter
venv-bench/bin/python gate-kit/run_kit.py --engine-python venv-vllm/bin/python --workload 8b --gpu h100
```

The options you are most likely to need:

| Option | Meaning |
|---|---|
| `--gpu h200, h100 or mi300x` | Sets the number of sessions, so the load against your KV cache is about what it was in the paper. See [SETTINGS.md](SETTINGS.md). |
| `--sessions N` | Overrides that number. |
| `--workload 8b or 32b-long` | Qwen3-8B with ~8k-token starts, or Qwen3-32B with ~20k-token starts. The second does not fit on one H100 80 GB. |
| `--engine sglang` | Runs on SGLang. The paper ran this gate only on vLLM. |
| `--model` | A different model. The sizes in SETTINGS.md then no longer hold. |
| `--capacity TOKENS` | Uses this KV capacity instead of the engine's own, for example when you size for a host-memory tier. |

The kit downloads the model the first time the engine starts.

It runs the baseline (the engine's own prefix cache), then the gate, then prints a table. Everything goes into
`gate-kit-out/`:
- **`gate-kit-results.json`:** the small results file to send us;
- `gate-kit-raw.json`: every request;
- `gate-kit-raw-timeline.json`: energy and engine counters over time;
- the engine logs.

### What is measured

For each of the two runs (baseline, then gate):
- **Load:** sessions started and sessions served (with at least one turn), and turns completed.
- **Cache-hit rate:** the share of prompt tokens the engine reported as cached.
- **Latency:** p50 and p90 of first-turn latency and of resume latency. Both include any wait for a place.
- **GPU energy per turn.**

For the gate run only:
- the KV capacity it used;
- the peak working set over the cache;
- how often it was switched on.

The percentiles are computed exactly as the paper computes them. On the paper's own runs, `summarize.py` gives the
paper's numbers: `test/test_kit.sh` checks this.

### Energy, and AMD

Energy is GPU board energy read by joules, the same meter and version as the paper.

- **NVIDIA:** the NVML energy counter, as in the paper.
- **AMD (MI300X):** joules reads the amdgpu energy counter (`energy1_input`) where the driver exposes it, and
  otherwise samples power (`power1_average`). This is not the meter the paper used. Compare it only with your own
  baseline from the same run.
- **No meter found:** the results file says *"not measured"* for energy, and the other numbers stand.

On AMD the kit answers the benchmark's `nvidia-smi` calls (free memory, temperature) from `rocm-smi`. We have not run
it on an MI300X; if `rocm-smi` answers differently, those waits are skipped and the run goes on. The benchmark picks the GPU with
`CUDA_VISIBLE_DEVICES` (from `--gpu-index`); if your ROCm build of vLLM needs `HIP_VISIBLE_DEVICES`, set that as well.

## Send us your results

Please email **`gate-kit-out/gate-kit-results.json`** to **bharat@wattops.dev**. That one file is enough. Add your GPU,
engine version and anything unusual about the machine. If something failed, add the engine log from `gate-kit-out/`.

The file holds:
- the numbers above;
- your GPU's name, driver and power limit;
- the operating system and Python version;
- the engine version.

It holds no prompts of yours: the sessions are synthetic.

What we do with it:
- We compare it with the paper and with other people's runs.
- **If we use your numbers in a write-up, we credit you** (your name, your company, or neither, as you prefer).
- **We do not publish or share your file or your name without your permission.**
- We will tell you what we found, including when your numbers disagree with the paper's.

## Test the kit without a GPU

```bash
bash gate-kit/test/test_kit.sh
```

The test runs in about 3 minutes and uses the standard library only. It checks four things:

1. The summarizer, run on the paper's own 10-minute results, gives the paper's numbers.
2. The H200 sizes are the paper's.
3. The whole kit runs end to end on the stand-in engine (`code/fake`, vLLM and SGLang), with a stand-in meter.
4. The AMD stand-in reads `rocm-smi` correctly.

## Files

| File | What |
|---|---|
| `run_kit.py` | Writes the two policies (baseline, gate) as a variants file, runs the paper's benchmark, summarizes. |
| `_agent_ctrl.py` | Runs `../code/agent_ctrl.py` unchanged, except that it keeps the results in the output folder instead of also copying them to `/`, as the study's rented machines did. |
| `summarize.py` | The results file and the table, from the benchmark's output. |
| `sizes.py` | Sessions per GPU at the paper's load against the cache (see SETTINGS.md). |
| `amd/nvidia-smi` | The `rocm-smi` stand-in, used only when there is no `nvidia-smi`. |
| `test/` | The test above and its stand-ins (a fake meter, a fake `nvidia-smi`). They are never used in a real run. |
| `SETTINGS.md` | Every setting and version of the paper's runs, and the sizes for other GPUs. |

Licence: as the rest of this repository ([LICENSE](../LICENSE)).
