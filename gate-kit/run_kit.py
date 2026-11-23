"""Run the paper's adaptive admission gate against the plain GPU cache on your own GPU, then write gate-kit-results.json.

    python3 gate-kit/run_kit.py --engine-python /path/to/venv-vllm/bin/python --workload 8b --gpu h100

It runs the paper's own benchmark (code/agent_ctrl.py) once with each policy, one after the other: the same synthetic
agent sessions (an ~8k- or ~20k-token start, a tool result each turn, 5 s median pauses), the same engine command, the
same gate and its settings (code/variants-p9-8b.json and code/variants-p9-lc.json). Then gate-kit/summarize.py writes
the small results file. Nothing in code/ is changed.

Needs: the engine in its own Python environment (vLLM 0.33.0 is what the paper ran), and this script run with a Python
that has `landauer-gap==0.8.2` (joules, the energy meter the paper used). About 25-30 minutes with the default 600 s steps.
"""
import argparse, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
import sizes, summarize  # noqa: E402

# The paper's 10-minute runs (code/phase9.sh and the two variants files): everything but the session count, which
# gate-kit/sizes.py scales to the card's KV room.
WORKLOADS = {
    "8b": {"context": 8000, "admit": {"mode": "adaptive", "turns": 4, "safety": 0.9, "growth": 600, "interval": 2, "theta": 1.2}},
    "32b-long": {"context": 20000, "admit": {"mode": "adaptive", "turns": 4, "safety": 0.9, "growth": 600, "interval": 2, "stat": "mean", "theta": 1.2}},
}


def amd_host():
    return not shutil.which("nvidia-smi") and bool(shutil.which("rocm-smi") or shutil.which("amd-smi"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm", help="the paper's gate runs were on vLLM")
    ap.add_argument("--engine-python", required=True, help="the Python of the environment the engine is installed in")
    ap.add_argument("--workload", choices=list(WORKLOADS), default="8b")
    ap.add_argument("--gpu", choices=list(sizes.GPUS), default="h200", help="sets the session count (gate-kit/sizes.py)")
    ap.add_argument("--sessions", type=int, help="override the session count")
    ap.add_argument("--model", help="override the model (the paper: Qwen/Qwen3-8B, Qwen/Qwen3-32B)")
    ap.add_argument("--duration", type=float, default=600, help="seconds per policy (the paper: 600)")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--capacity", type=int, help="KV capacity in tokens, instead of asking the engine (for a host tier, or tests)")
    ap.add_argument("--out-dir", default="gate-kit-out")
    # the rest are for tests on the stand-in engine; leave them at the paper's values for a real run
    for flag, default in (("--tool-tokens", 400), ("--answer-tokens", 200), ("--idle", 30), ("--cooldown", 20),
                          ("--settle-window", 60), ("--startup-timeout", 2400), ("--max-len", None), ("--pauses", "5")):
        ap.add_argument(flag, default=default)
    a = ap.parse_args()

    wl, model = WORKLOADS[a.workload], a.model or sizes.WORKLOADS[a.workload]["model"]
    n = a.sessions or sizes.size(a.workload, sizes.GPUS[a.gpu])["sessions"]
    if not n:
        sys.exit(f"{a.workload} does not fit on one {a.gpu} (see gate-kit/sizes.py); choose the other workload or --sessions.")
    try:
        import joules  # noqa: F401
    except ImportError:
        sys.exit("joules is not installed in this Python: pip install landauer-gap==0.8.2")
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.abspath(a.out_dir)
    admit = dict(wl["admit"], **({"capacity": a.capacity} if a.capacity else {}))
    base = {"engine": a.engine, "python": a.engine_python, "flags": []}
    variants = [dict(base, name=f"{a.engine}-gpu-{a.workload}", policy="gpu-only"),
                dict(base, name=f"{a.engine}-gate-{a.workload}", policy="brain-v0-adaptive", admit=admit)]
    vf = os.path.join(out, "variants.json")
    json.dump(variants, open(vf, "w"), indent=1)

    env = dict(os.environ)
    if amd_host():   # the paper's scripts ask nvidia-smi for free memory and temperature; answer them from rocm-smi
        env["PATH"] = os.path.join(HERE, "amd") + os.pathsep + env.get("PATH", "")
        print("AMD GPU: using gate-kit/amd/nvidia-smi (a stand-in that asks rocm-smi)", flush=True)
    raw = os.path.join(out, "gate-kit-raw.json")
    cmd = [sys.executable, os.path.join(HERE, "_agent_ctrl.py"), "--model", model, "--variants", vf,
           "--sessions", str(n), "--pauses", str(a.pauses), "--duration", str(a.duration), "--context", str(wl["context"]),
           "--max-len", str(a.max_len or sizes.WORKLOADS[a.workload]["max_len"]), "--rounds", "1", "--cpu",
           "--gpu", str(a.gpu_index), "--port", str(a.port), "--tool-tokens", str(a.tool_tokens),
           "--answer-tokens", str(a.answer_tokens), "--idle", str(a.idle), "--cooldown", str(a.cooldown),
           "--settle-window", str(a.settle_window), "--startup-timeout", str(a.startup_timeout), "--out", raw]
    print(f"{model}, {n} sessions, {a.duration:.0f} s per policy, {a.engine}: baseline, then the gate\n" + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=out, env=env)
    if not os.path.exists(raw):
        sys.exit(f"the benchmark stopped before writing results (exit {r.returncode}); see the server logs in {out}")
    S = summarize.summarize(json.load(open(raw)))
    S["kit_run"] = {"workload": a.workload, "gpu_preset": a.gpu, "sessions": n, "engine_python": a.engine_python,
                    "amd_stand_in": amd_host(), "capacity_given": a.capacity}
    res = os.path.join(out, "gate-kit-results.json")
    json.dump(S, open(res, "w"), indent=1)
    print("\n" + summarize.table(S) + f"\n\nResults: {res}\nPlease email that one file (see gate-kit/README.md).")
    return 0 if r.returncode == 0 and not S["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
