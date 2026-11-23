"""How many sessions put another GPU at the same load relative to its KV cache as the paper's 10-minute runs.

    python3 -I gate-kit/sizes.py              # the table in gate-kit/SETTINGS.md
    python3 -I gate-kit/sizes.py --gib 94     # another card: its memory in GiB as the driver reports it

Sizes only, no new claims. The paper's runs (one H200 NVL, vLLM 0.33.0, --gpu-memory-utilization 0.90) logged their KV
capacity: 777,872 tokens for Qwen3-8B and 244,928 for Qwen3-32B (results/engine-log-lines). Everything else the engine
keeps (weights, activations, CUDA graphs) is taken as the same on the other card; the KV room left is scaled, and the
session count with it, so the working set over the cache is about what it was in the paper. The engine's own figure
("GPU KV cache size" in the server log, and kv_capacity_tokens in the kit's results) is the one to trust.
"""
import argparse, math, sys

GIB = 1 << 30
H200_GIB = 143771 / 1024          # memory.total of the paper's H200 NVL, MiB -> GiB
UTIL = 0.90                       # --gpu-memory-utilization in the paper's vLLM command
# per workload: model, KV bytes per token (2 x layers x KV heads x head dim x 2 bytes, BF16), the paper's KV capacity
# on the H200 in tokens, the paper's session count, and the longest request it allowed (--max-len)
WORKLOADS = {
    "8b": {"model": "Qwen/Qwen3-8B", "kv_bytes": 2 * 36 * 8 * 128 * 2, "c_tokens": 777_872, "sessions": 256, "max_len": 32768},
    "32b-long": {"model": "Qwen/Qwen3-32B", "kv_bytes": 2 * 64 * 8 * 128 * 2, "c_tokens": 244_928, "sessions": 32, "max_len": 40960},
}
GPUS = {"h200": H200_GIB, "h100": 81559 / 1024, "mi300x": 192.0}   # GiB; H100 80 GB SXM/PCIe; MI300X 192 GB (check rocm-smi)


def size(workload, gib):
    w = WORKLOADS[workload]
    kv_h200 = w["c_tokens"] * w["kv_bytes"] / GIB
    other = UTIL * H200_GIB - kv_h200          # what the engine kept for everything but KV on the H200
    kv = UTIL * gib - other
    tokens = int(kv * GIB / w["kv_bytes"]) if kv > 0 else 0
    if tokens < w["max_len"]:
        return {"kv_tokens": tokens, "sessions": None, "note": "does not fit: one full-length session needs more KV room than this card has left"}
    return {"kv_tokens": tokens, "sessions": max(1, round(w["sessions"] * tokens / w["c_tokens"])), "note": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, help="GPU memory in GiB (default: the H200, H100 80 GB and MI300X table)")
    a = ap.parse_args()
    cards = {"this card": a.gib} if a.gib else GPUS
    print("| Workload | GPU | KV room (tokens, estimate) | Sessions |")
    print("|---|---|---|---|")
    for wl in WORKLOADS:
        for g, gib in cards.items():
            r = size(wl, gib)
            print(f"| {wl} | {g} | {r['kv_tokens']:,} | {r['sessions'] if r['sessions'] else r['note']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
