"""Does keeping KV in GPU memory cost power? Idle GPU power with an empty prefix cache vs a full one.

Per engine: start it, let the card settle, measure idle power (empty cache); fill the cache to about --fill of its
capacity with distinct prompts (one output token each); let the card settle again and measure idle power (full
cache); measure once more after a second settle. Each measurement is a joules phase on the NVML energy counter.

    python idle_cache.py --model Qwen/Qwen3-32B --variants variants-idle.json --capacity 244928
"""
import argparse, json, os, random, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_energy2 import ask, base_cmd, document, gpu_j, gpu_temp, settle, stop, wait_ready  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--capacity", type=int, required=True, help="KV capacity in tokens (from the engine log)")
    ap.add_argument("--fill", type=float, default=0.85)
    ap.add_argument("--prompt-tokens", type=int, default=8000)
    ap.add_argument("--measure", type=float, default=90)
    ap.add_argument("--settle-window", type=float, default=60)
    ap.add_argument("--startup-timeout", type=float, default=2400)
    ap.add_argument("--out", default="idle-cache.json")
    a = ap.parse_args()
    import joules
    meter = joules.Session(gpus=[a.gpu], interval=0.1)
    R = {"schema": "idle-cache/1", "model": a.model, "settings": vars(a), "meter": meter.devices, "runs": []}
    for v in json.load(open(a.variants)):
        cmd = base_cmd(v["engine"], v["python"], a.model, a.port, a.max_len) + v.get("flags", [])
        E = {"variant": v["name"], "command": cmd, "phases": []}
        R["runs"].append(E)
        print(f"\n=== {v['name']}\n{' '.join(cmd)}", flush=True)
        log = os.path.abspath(f"{v['name']}-idle-server.log")
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)), start_new_session=True)
        try:
            wait_ready(a.port, proc, a.startup_timeout, log)
            ask(a.port, "Warm-up.\n" + document(500, random.Random(0)), 4)

            def measure(label):
                st = settle(a.gpu, a.settle_window)
                with meter.phase(f"idle:{v['name']}:{label}") as ph:
                    time.sleep(a.measure)
                w = round(gpu_j(ph, meter) / ph.duration_s, 3)
                E["phases"].append({"label": label, "idle_w": w, "seconds": round(ph.duration_s, 1), "settle": st,
                                    "temp_c": gpu_temp(a.gpu)})
                print(f"  {label:12s}: {w} W (settle {st})", flush=True)

            measure("empty")
            n = int(a.fill * a.capacity / a.prompt_tokens)
            ps = [f"Fill {v['name']}-{i}.\n\n" + document(a.prompt_tokens, random.Random(f"fill-{i}")) for i in range(n)]
            cached = []
            for k in range(0, n, 8):   # 8 at a time, one output token each
                th = [threading.Thread(target=lambda p=p: cached.append(ask(a.port, p, 1)["prompt"])) for p in ps[k:k + 8]]
                [t.start() for t in th]; [t.join() for t in th]
            E["filled_prompt_tokens"] = sum(x or 0 for x in cached)
            check = ask(a.port, ps[0], 1)     # the first fill prompt should still be cached if nothing was evicted
            E["first_fill_prompt_cached_frac"] = round((check["cached"] or 0) / check["prompt"], 4) if check["prompt"] else None
            print(f"  filled ~{E['filled_prompt_tokens']} prompt tokens; first fill prompt cached {E['first_fill_prompt_cached_frac']}", flush=True)
            measure("full")
            measure("full-again")
        except Exception as e:  # noqa: BLE001
            E["error"] = repr(e)
            print(f"!! {v['name']}: {e!r}", flush=True)
        finally:
            stop(proc)
            time.sleep(10)
            json.dump(R, open(a.out, "w"), indent=1)
    meter.close()
    print(f"=== saved {a.out}")


if __name__ == "__main__":
    main()
