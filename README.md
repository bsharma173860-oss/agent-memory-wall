# The Agent Memory Wall: artifact

Code, results and pre-registration record for the study *The Agent Memory Wall: Measuring and Managing the GPU Energy
of KV Caches for AI Agents* (Bharat Bhushan Sharma, PHENEX TECHNOLOGIES INC / WattOps). The paper is in [`paper/agent-memory-wall.pdf`](paper/agent-memory-wall.pdf);
the arXiv link will be added here when it is on arXiv.

The study measures the GPU energy of KV-cache policies for AI agents on one NVIDIA H200 NVL (the 10-minute adaptive-gate
runs of phase 9 used a second H200 NVL host) with vLLM 0.33 and SGLang 0.5.23, three Qwen3 models, synthetic agent sessions, recorded SWE-agent trajectories, a replay of Kimi's
production tool-agent trace, and the BFCL v4 multi-turn benchmark. All energy figures are GPU board energy read from
the NVML energy counter.

## Contents

| Folder | What |
|---|---|
| `code/` | Benchmark scripts (`agent_*.py`, `mooncake_replay.py`, `transfer_energy.py`, `bfcl_energy.py`, ...), the policy files (`variants-*.json`) and the run scripts in the order they ran (`run_all.sh`, `extra.sh`, `phase2.sh` ... `phase8.sh`, `rest.sh`, then `setup9.sh` and `phase9.sh` on the second host). `fake/` is a fake OpenAI-compatible engine used to test every script end to end before GPU time was rented. |
| `results/` | One archive per run block, JSON and CSV only (per-step results, per-request rows, energy and engine-counter timelines). Engine logs and copies of third-party datasets are not included; `results/engine-log-lines/` keeps, unchanged, the few log lines the paper's numbers depend on (KV capacities, weight size, rejected BFCL requests). |
| `analysis/` | The analysis script, tables and figures built from `results/`. |
| `paper/` | The paper's LaTeX sources (`tex/`), bibliography and figures; `make_figures.py` rebuilds the figures from `results/`; `verify_numbers.py` recomputes every number printed in the paper from `results/` and checks it against the text (742 checks); `build_arxiv.sh` builds the arXiv package. |
| `prereg/` | The study plan with every prediction (H1-H28), as it stood at each of its 51 commits, with the SHA-256 of each version (`INDEX.md`), plus the theory notes. |

## Pre-registration

Every hypothesis was written into the plan before the runs that tested it, with a numeric threshold. Verdicts were
appended in later commits and no prediction was edited. `prereg/INDEX.md` lists each version of the plan with its
commit time (UTC) and hash; the paper's Appendix A maps each hypothesis to the commit that added it. Some predictions
were added during the study, after earlier runs; the plan says which results had been seen when each was written. One
block of runs was voided after two copies of a run script shared the GPU; the plan records the incident and none of
its results are used.

## Reproducing

1. Rent one H200 (or similar) with large host memory; run `code/setup.sh` to create the engine and benchmark
   environments and download the model.
2. Fetch the inputs we do not redistribute:
   - Production trace: `code/fetch_traces.sh` (Mooncake FAST'25 release, tool-agent trace; checked by SHA-256).
   - SWE-agent trajectories: the 400 trajectories come from the public `nebius/SWE-agent-trajectories` dataset on
     Hugging Face; `code/swe_traj_ids.txt` lists their instance ids. Our converted file (`{id, resolved, system,
     messages}` per line) had SHA-256 `a7b8f35465b7500b2f06522be73d516badb3f6cd35598e525aa30c4b941fee51`.
   - BFCL v4: install `bfcl-eval`; `bfcl_energy.py` drives its own harness and grader.
3. Run the phase scripts. Each script states which hypotheses it tests; every engine start waits for a free GPU.
   `agent_replay.py` now gives an explicitly configured capacity (e.g. a host tier's) precedence over the engine's
   reported one; the H21 runs used the reverse order and were therefore sized to the GPU, as the paper reports.

To check the paper against the data without a GPU: `python3 -I paper/verify_numbers.py` (standard library only; it
prints every check and exits non-zero if any printed number disagrees with `results/`). `python3 paper/make_figures.py`
rebuilds the three data figures (needs matplotlib).

To check a script without a GPU, put `code/fake` first on `PYTHONPATH` and use a variants file whose `python` is your
local interpreter: the fake engine streams tokens, keeps an LRU prefix cache and reports cached tokens.

## Data sources and credit

- Mooncake / Kimi tool-agent trace (`toolagent_trace.jsonl`, FAST'25 release): Qin et al., *Mooncake: Trading More
  Storage for Less Computation -- A KVCache-centric Architecture for Serving LLM Chatbot*, FAST '25,
  https://github.com/kvcache-ai/Mooncake
- SWE-agent trajectories: Nebius, https://huggingface.co/datasets/nebius/SWE-agent-trajectories
- BFCL: Patil et al., *The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large
  Language Models*, ICML 2025, https://gorilla.cs.berkeley.edu/leaderboard.html
- Models: Qwen3 (arXiv:2505.09388)

## Licenses

Code: Apache License 2.0 (`LICENSE`). Results, analysis and pre-registration documents: CC BY 4.0 (`LICENSE-DATA`).
Third-party datasets keep their own licenses and are not included here.

## Citation

Please cite the paper (BibTeX will be added with the arXiv link).
