"""Does a cached prompt expire with time when nothing else needs the memory?

Per engine: send N prompts of ~L tokens, then resend prompt i once after delays[i] seconds of near-idle (no memory
pressure: N x L is far below KV capacity). If the cache were time-based, late resends would come back uncached.
Records cached tokens and time to first token for every send.

    python retention_test.py --model Qwen/Qwen3-32B --variants variants-retention.json --delays 0,30,120,300,600
"""
import argparse, json, os, random, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_energy2 import ask, base_cmd, document, stop, wait_ready  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--length", type=int, default=8000)
    ap.add_argument("--delays", default="0,30,120,300,600")
    ap.add_argument("--startup-timeout", type=float, default=2400)
    ap.add_argument("--out", default="retention.json")
    a = ap.parse_args()
    delays = [float(x) for x in a.delays.split(",")]
    R = {"schema": "retention/1", "model": a.model, "settings": vars(a), "runs": []}
    for v in json.load(open(a.variants)):
        cmd = base_cmd(v["engine"], v["python"], a.model, a.port, a.max_len) + v.get("flags", [])
        E = {"variant": v["name"], "command": cmd, "sends": []}
        R["runs"].append(E)
        print(f"\n=== {v['name']}\n{' '.join(cmd)}", flush=True)
        log = os.path.abspath(f"{v['name']}-retention-server.log")
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)), start_new_session=True)
        try:
            wait_ready(a.port, proc, a.startup_timeout, log)
            ps = [f"Retention probe {v['name']}-{i}.\n\n" + document(a.length, random.Random(f"ret-{i}")) for i in range(len(delays))]
            lock = threading.Lock()

            def send(i, kind, d):
                r = ask(a.port, ps[i], 1); r.pop("text", None)
                row = {"prompt_id": i, "send": kind, "delay_s": d, "prompt_tokens": r["prompt"], "cached_tokens": r["cached"],
                       "cached_frac": round((r["cached"] or 0) / r["prompt"], 4) if r["prompt"] else None, "ttft_s": round(r["ttft"], 4)}
                with lock:
                    E["sends"].append(row)
                print(f"  prompt {i} {kind:6s} after {d:5.0f} s: cached {row['cached_frac']}, ttft {row['ttft_s']} s", flush=True)

            for i in range(len(delays)):
                send(i, "first", 0)
            t0 = time.monotonic()
            for i, d in sorted(enumerate(delays), key=lambda x: x[1]):
                time.sleep(max(0.0, t0 + d - time.monotonic()))
                send(i, "resend", d)
        except Exception as e:  # noqa: BLE001
            E["error"] = repr(e)
            print(f"!! {v['name']}: {e!r}", flush=True)
        finally:
            stop(proc)
            time.sleep(10)
            json.dump(R, open(a.out, "w"), indent=1)
    print(f"=== saved {a.out}")


if __name__ == "__main__":
    main()
