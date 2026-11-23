"""Correct answers per kWh: BFCL multi-turn (an agent benchmark with automatic grading) under different engine
cache policies, with GPU energy metered over each whole generation run.

Cache policies are lossless, so accuracy should not change; what changes is GPU energy and time per task. Each run:
start the engine (served under both 'bench' and the HF id BFCL sends), settle, measure idle, then for each round run
`bfcl generate` against it inside a joules phase, stop the engine, and grade with `bfcl evaluate`.

    python bfcl_energy.py --variants variants-bfcl.json --categories multi_turn_base,multi_turn_long_context \
        --threads 100 --rounds 2 --out bfcl.json
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_energy2 import base_cmd, gpu_j, settle, stop, wait_ready  # noqa: E402

BFCL = "/workspace/venv-bfcl/bin/bfcl"


def scores(score_dir):
    """Every row of BFCL's score CSVs, as found (format varies by version), plus per-category accuracy if present."""
    out = {}
    for f in glob.glob(os.path.join(score_dir, "**", "*.csv"), recursive=True):
        with open(f) as fh:
            out[os.path.relpath(f, score_dir)] = list(csv.reader(fh))
    acc = {}
    for f in glob.glob(os.path.join(score_dir, "**", "*_score.json"), recursive=True):
        try:
            first = json.loads(open(f).readline())
            acc[os.path.basename(f)] = {k: first.get(k) for k in ("accuracy", "correct_count", "total_count")}
        except Exception:  # noqa: BLE001
            pass
    return out, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--bfcl-model", default="Qwen/Qwen3-32B-FC")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--categories", default="multi_turn_base,multi_turn_long_context")
    ap.add_argument("--threads", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--idle", type=float, default=30)
    ap.add_argument("--settle-window", type=float, default=60)
    ap.add_argument("--startup-timeout", type=float, default=2400)
    ap.add_argument("--out", default="bfcl.json")
    a = ap.parse_args()
    import joules
    meter = joules.Session(gpus=[a.gpu], interval=0.1)
    R = {"schema": "bfcl-energy/1", "settings": vars(a), "meter": meter.devices, "runs": []}
    variants = json.load(open(a.variants))
    for rnd in range(1, a.rounds + 1):
        for v in (variants if rnd % 2 else variants[::-1]):
            name = v["name"]
            cmd = base_cmd(v["engine"], v["python"], a.model, a.port, a.max_len) + v.get("flags", [])
            if "--disable-prefix-caching" in cmd:
                cmd = [c for c in cmd if c != "--enable-prefix-caching"]
            cmd += ["--served-model-name", "bench", a.model]          # the last flag wins: both names are served
            E = {"variant": name, "round": rnd, "command": cmd}
            R["runs"].append(E)
            print(f"\n=== {name} (round {rnd})\n{' '.join(cmd)}", flush=True)
            log = os.path.abspath(f"{name}-bfcl-r{rnd}-server.log")
            res = os.path.abspath(f"bfcl-results/{name}-r{rnd}"); sc = os.path.abspath(f"bfcl-scores/{name}-r{rnd}")
            shutil.rmtree(res, ignore_errors=True); shutil.rmtree(sc, ignore_errors=True)
            os.makedirs(os.path.dirname(res), exist_ok=True); os.makedirs(os.path.dirname(sc), exist_ok=True)
            t_wait = time.time()   # the previous engine must have released the GPU (it OOMed once without this)
            while time.time() - t_wait < 900:
                used = subprocess.run(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True).stdout.strip()
                if used.isdigit() and int(used) < 2000:
                    break
                time.sleep(10)
            with open(log, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)), start_new_session=True)
            try:
                E["startup_s"] = round(wait_ready(a.port, proc, a.startup_timeout, log), 1)
                E["settle"] = settle(a.gpu, a.settle_window)
                with meter.phase(f"bfcl:{name}:r{rnd}:idle") as ph:
                    time.sleep(a.idle)
                E["idle_w"] = round(gpu_j(ph, meter) / ph.duration_s, 2)
                env = dict(os.environ, LOCAL_SERVER_ENDPOINT="127.0.0.1", LOCAL_SERVER_PORT=str(a.port))
                with meter.phase(f"bfcl:{name}:r{rnd}") as ph:
                    g = subprocess.run([BFCL, "generate", "--model", a.bfcl_model, "--test-category", a.categories,
                                        "--skip-server-setup", "--num-threads", str(a.threads), "--result-dir", res,
                                        "--allow-overwrite"], env=env, capture_output=True, text=True, cwd=os.path.dirname(res))
                E["generate_rc"], E["generate_tail"] = g.returncode, (g.stdout + g.stderr)[-3000:]
                E["energy_j"], E["wall_s"] = round(gpu_j(ph, meter), 1), round(ph.duration_s, 1)
                E["avg_w"] = round(E["energy_j"] / E["wall_s"], 1)
                print(f"  generate rc {g.returncode}: {E['energy_j']} J over {E['wall_s']} s ({E['avg_w']} W)", flush=True)
            except Exception as e:  # noqa: BLE001
                E["error"] = repr(e); print(f"!! {name}: {e!r}", flush=True)
            finally:
                stop(proc); time.sleep(10)
            ev = subprocess.run([BFCL, "evaluate", "--model", a.bfcl_model, "--test-category", a.categories,
                                 "--result-dir", res, "--score-dir", sc], capture_output=True, text=True, cwd=os.path.dirname(res))
            E["evaluate_rc"], E["evaluate_tail"] = ev.returncode, (ev.stdout + ev.stderr)[-3000:]
            E["score_csv"], E["accuracy"] = scores(sc)
            print(f"  evaluate rc {ev.returncode}: {E['accuracy']}", flush=True)
            json.dump(R, open(a.out, "w"), indent=1)
    meter.close()
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"=== saved {a.out}")


if __name__ == "__main__":
    main()
