"""Transfer or recompute? GPU energy per prompt token of the three ways an engine can get a prompt's KV cache:
a prefix-cache hit on the GPU, a restore from the host tier, or a full recompute (prefill).

For each prompt length and repetition: send k fresh prompts at once (recompute), send them again (GPU hit), push
them out of GPU memory by prefilling filler prompts (unmeasured), and send them a third time (restore if the engine
has a host tier that still holds them; recompute otherwise, which makes a GPU-only engine the control). Each send
asks for one output token, so the measured energy is almost all prefill or restore. Gross and above-idle energy are
both reported, with cached tokens and time to first token, so the path each send took can be checked.

    python transfer_energy.py --model Qwen/Qwen3-32B --variants variants-transfer.json --lengths 4000,8000,16000 \
        --k 8 --flush-tokens 300000 --reps 3 --out transfer.json
"""
import argparse, json, os, platform, random, shutil, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_energy2 import ask, base_cmd, check_flags, document, gpu_j, pct, settle, stop, wait_ready  # noqa: E402


def send_all(port, prompts, max_tokens=1):
    out = [None] * len(prompts)

    def go(i):
        try:
            r = ask(port, prompts[i], max_tokens); r.pop("text", None); out[i] = r
        except Exception as e:  # noqa: BLE001
            out[i] = {"error": repr(e)[:200]}
    th = [threading.Thread(target=go, args=(i,), daemon=True) for i in range(len(prompts))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def prompts_for(tag, n, length):
    return [f"Request {tag}-{i:03d}. Review the code below.\n\n" + document(length, random.Random(f"{tag}-{i}")) for i in range(n)]


def measure(meter, key, port, prompts, idle_w):
    with meter.phase(key) as ph:
        r = send_all(port, prompts)
    ok = [x for x in r if "error" not in x]
    gj, dur = gpu_j(ph, meter), ph.duration_s
    pt = sum(x["prompt"] or 0 for x in ok); ct = sum(x["cached"] or 0 for x in ok)
    above = gj - idle_w * dur
    return {"requests": len(prompts), "failed": len(prompts) - len(ok), "errors": [x["error"] for x in r if "error" in x][:3],
            "prompt_tokens": pt, "cached_tokens": ct, "cached_frac": round(ct / pt, 4) if pt else None,
            "energy_j": round(gj, 2), "duration_s": round(dur, 3), "above_idle_j": round(above, 2),
            "j_per_prompt_token": round(gj / pt, 5) if pt else None,
            "above_idle_j_per_prompt_token": round(above / pt, 5) if pt else None,
            "ttft_p50": pct([x["ttft"] for x in ok], .5), "ttft_max": pct([x["ttft"] for x in ok], 1.0)}


def run_variant(v, a, meter, R):
    name = v["name"]
    log = os.path.abspath(f"{name}-transfer-server.log")
    cmd = base_cmd(v["engine"], v["python"], a.model, a.port, a.max_len) + v.get("flags", [])
    E = {"variant": name, "engine": v["engine"], "flags": v.get("flags", []), "command": cmd, "steps": []}
    R["runs"].append(E)
    print(f"\n=== {name}\n{' '.join(cmd)}", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)),
                                start_new_session=True)
    try:
        E["startup_s"] = round(wait_ready(a.port, proc, a.startup_timeout, log), 1)
        for i in range(4):
            ask(a.port, f"Warm-up {i}.\n" + document(2000, random.Random(i)), 16)
        E["settle"] = settle(a.gpu, a.settle_window)
        with meter.phase(f"transfer:{name}:idle") as ph:
            time.sleep(a.idle)
        idle_w = E["idle_w"] = round(gpu_j(ph, meter) / ph.duration_s, 2)
        print(f"ready in {E['startup_s']} s, idle {idle_w} W", flush=True)
        for rep in range(1, a.reps + 1):
            for L in a.lengths:
                tag = f"{name}-r{rep}-L{L}"
                ps = prompts_for(tag, a.k, L)
                s = {"rep": rep, "length": L}
                s["recompute"] = measure(meter, f"transfer:{tag}:recompute", a.port, ps, idle_w)
                time.sleep(a.gap)
                s["gpu_hit"] = measure(meter, f"transfer:{tag}:gpu_hit", a.port, ps, idle_w)
                fill, n, j = 0, 0, 0
                while fill < a.flush_tokens:   # push the prompts out of GPU memory (unmeasured)
                    fs = prompts_for(f"{tag}-fill{j}", a.flush_batch, a.flush_len)
                    r = send_all(a.port, fs)
                    fill += sum((x.get("prompt") or 0) for x in r if "error" not in x) or a.flush_batch * a.flush_len
                    j += 1
                s["flush_tokens"] = fill
                time.sleep(a.gap)
                s["after_flush"] = measure(meter, f"transfer:{tag}:after_flush", a.port, ps, idle_w)
                E["steps"].append(s)
                f = lambda k: s[k]["above_idle_j_per_prompt_token"]
                print(f"  rep {rep}, ~{L} tokens x {a.k}: J/token above idle: recompute {f('recompute')}, GPU hit "
                      f"{f('gpu_hit')}, after flush {f('after_flush')} (cached {s['after_flush']['cached_frac']}, "
                      f"ttft {s['after_flush']['ttft_p50']} s vs recompute {s['recompute']['ttft_p50']} s)", flush=True)
                json.dump(R, open(a.out, "w"), indent=1)
                time.sleep(a.gap)
    except Exception as e:  # noqa: BLE001
        E["error"] = repr(e)
        print(f"!! {name}: {e!r}", flush=True)
    finally:
        stop(proc)
        time.sleep(10)
        json.dump(R, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--lengths", default="4000,8000,16000", help="approx prompt tokens")
    ap.add_argument("--k", type=int, default=8, help="prompts sent at once per measurement")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--flush-tokens", type=int, default=300000, help="filler tokens to push the prompts out of GPU memory")
    ap.add_argument("--flush-len", type=int, default=16000)
    ap.add_argument("--flush-batch", type=int, default=16)
    ap.add_argument("--gap", type=float, default=3, help="seconds between measured sends")
    ap.add_argument("--idle", type=float, default=30)
    ap.add_argument("--settle-window", type=float, default=60)
    ap.add_argument("--startup-timeout", type=float, default=2400)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="transfer.json")
    a = ap.parse_args()
    a.lengths = [int(x) for x in a.lengths.split(",")]
    variants = json.load(open(a.variants))
    R = {"schema": "transfer-energy/1", "model": a.model, "settings": vars(a), "variants": variants, "runs": [], "skipped": {},
         "host": {"os": platform.platform(), "python": platform.python_version()},
         "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for v in variants:
        missing = check_flags(v)
        if missing:
            R["skipped"][v["name"]] = f"flags not in this engine's --help: {missing}"
            print(f"!! skipping {v['name']}: flags not found {missing}", flush=True)
    variants = [v for v in variants if v["name"] not in R["skipped"]]
    import joules
    meter = joules.Session(gpus=[a.gpu], cpu=a.cpu, interval=0.1)
    R["joules"], R["meter"] = joules.__version__, meter.devices
    print(f"meter: {meter.devices}\nvariants: {[v['name'] for v in variants]}", flush=True)
    for v in variants:
        run_variant(v, a, meter, R)
    R["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meter.close()
    json.dump(R, open(a.out, "w"), indent=1)
    try:
        shutil.copy(a.out, os.path.join("/", os.path.basename(a.out)))
    except OSError:
        pass
    print(f"=== saved {a.out}")


if __name__ == "__main__":
    main()
