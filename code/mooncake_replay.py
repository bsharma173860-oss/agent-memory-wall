"""E8: production agent traffic. GPU energy of KV-cache policies on a real production tool-agent trace.

Trace: Mooncake (Kimi, Moonshot AI) FAST'25 release, traces/toolagent_trace.jsonl: one hour of production tool-agent
requests, each with its arrival time (ms), prompt length, output length and the hash of every 512-token prompt block.
Requests that share a prompt prefix share block hashes, so the trace keeps the real prefix sharing and timing while
the text itself is withheld. We rebuild each block as 512 fixed token ids derived from its hash: two requests share
exactly the prefix tokens the production requests shared, so a cache sees the same reuse. Energy depends on token
counts and reuse, not on the words.

One H200 cannot carry a cluster's traffic, so the trace is sampled by prefix group (the first block after the shared
system prompts): a sampled group keeps all of its requests, so conversations stay whole. Fractions are nested (the
1/16 sample is inside the 1/8 sample). Requests are sent open-loop at their recorded times (minus --offset), streamed,
with the recorded output length (ignore_eos). After the window, in-flight requests may finish for --drain seconds;
energy covers the whole window plus the drain.

    python mooncake_replay.py --model Qwen/Qwen3-32B --variants variants-mooncake.json --trace traces/toolagent_trace.jsonl \
        --fractions 0.0625,0.125,0.25 --window 300 --rounds 2 --cpu --out mooncake.json
"""
import argparse, collections, http.client, json, os, random, subprocess, sys, threading, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_energy2 import Recorder, ask, base_cmd, check_flags, document, gpu_j, gpu_temp, pct, settle, stop, wait_ready  # noqa: E402

BLOCK = 512
_tok_cache = {}


def block_tokens(h):
    """512 token ids for block hash h: fixed, ordinary vocabulary ids (Qwen3 special tokens start at 151643)."""
    t = _tok_cache.get(h)
    if t is None:
        rng = random.Random(1_000_003 * h + 17)
        t = _tok_cache[h] = [rng.randrange(1000, 150000) for _ in range(BLOCK)]
    return t


def prompt_ids(r):
    ids = []
    for h in r["hash_ids"]:
        ids.extend(block_tokens(h))
    return ids[:r["input_length"]]


def group_keys(R, popular=200):
    """Prefix group of each request: its first block hash used by fewer than `popular` requests (past the shared system
    prompts); a request made only of popular blocks is its own group."""
    cnt = collections.Counter(h for r in R for h in r["hash_ids"])
    keys = []
    for i, r in enumerate(R):
        k = next((h for h in r["hash_ids"] if cnt[h] < popular), None)
        keys.append(f"h{k}" if k is not None else f"r{i}")
    return keys


def sample(R, keys, frac, t_from, t_to, max_len):
    sel, skipped = [], 0
    for r, k in zip(R, keys):
        if not (t_from <= r["timestamp"] / 1000 < t_to):
            continue
        if zlib.crc32(k.encode()) % 1_000_000 >= frac * 1_000_000:
            continue
        if r["input_length"] + r["output_length"] > max_len:
            skipped += 1; continue
        sel.append(r)
    return sel, skipped


def lru_hit(reqs, cap_tokens):
    """Prediction: prefix hit rate with block LRU of cap_tokens (a radix cache is LRU on prefix blocks; a request
    reuses its blocks up to the first missing one)."""
    lru, hit, tot = collections.OrderedDict(), 0, 0
    nb = max(1, cap_tokens // BLOCK)
    for r in reqs:
        n = r["input_length"]; full = n // BLOCK
        prefix = True
        for j, h in enumerate(r["hash_ids"]):
            size = BLOCK if j < full else n - full * BLOCK
            if size <= 0:
                break
            if prefix and h in lru and j < full:
                hit += size
            else:
                prefix = False
            tot += size
            lru[h] = 1; lru.move_to_end(h)
            while len(lru) > nb:
                lru.popitem(last=False)
    return hit / tot if tot else 0.0


def complete(port, ids, n, timeout=1200):
    body = json.dumps({"model": "bench", "prompt": ids, "max_tokens": n, "temperature": 0.0, "ignore_eos": True,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    t0 = time.perf_counter(); ttft, usage = None, None
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("text") and ttft is None:
                    ttft = time.perf_counter() - t0
    finally:
        conn.close()
    usage = usage or {}
    return {"tokens": usage.get("completion_tokens") or 0, "prompt": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "ttft": ttft,
            "e2e": time.perf_counter() - t0}


def step(port, reqs, a, offset):
    out, lock = [], threading.Lock()
    payloads = [(r["timestamp"] / 1000 - offset, prompt_ids(r), max(1, min(a.max_output, r["output_length"]))) for r in reqs]

    def one(i, ids, n):
        try:
            x = complete(port, ids, n); x["i"] = i
        except Exception as e:  # noqa: BLE001
            x = {"i": i, "error": repr(e)[:200]}
        with lock:
            out.append(x)

    th = []
    t0 = time.perf_counter()
    for i, (t, ids, n) in enumerate(payloads):
        d = t0 + t - time.perf_counter()
        if d > 0:
            time.sleep(d)
        x = threading.Thread(target=one, args=(i, ids, n), daemon=True); x.start(); th.append(x)
    sent = time.perf_counter() - t0
    end = t0 + a.window + a.drain
    for x in th:
        x.join(max(0.0, end - time.perf_counter()))
    return out, sent, time.perf_counter() - t0


def run_variant(v, rnd, a, meter, R, samples):
    name = v["name"]
    log = os.path.abspath(f"{name}-mc-r{rnd}-server.log")
    cmd = base_cmd(v["engine"], v["python"], a.model, a.port, a.max_len) + v.get("flags", [])
    if "--disable-prefix-caching" in cmd or "--no-enable-prefix-caching" in cmd:
        cmd = [c for c in cmd if c != "--enable-prefix-caching"]
    E = {"variant": name, "engine": v["engine"], "policy": v.get("policy"), "round": rnd, "command": cmd, "steps": []}
    R["runs"].append(E)
    missing = check_flags(v)
    if missing:
        E["skipped"] = f"flags not in --help: {missing}"; print(f"!! {name}: {E['skipped']}", flush=True); return
    print(f"\n=== {name} (round {rnd})\n{' '.join(cmd)}", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)),
                                start_new_session=True)
    try:
        E["startup_s"] = round(wait_ready(a.port, proc, a.startup_timeout, log), 1)
        for i in range(4):
            ask(a.port, f"Warm-up {i}.\n" + document(2000, random.Random(i)), 16)
        E["settle"] = settle(a.gpu, a.settle_window)
        with meter.phase(f"mc:{name}:r{rnd}:idle") as ph:
            time.sleep(a.idle)
        E["idle_w"] = round(gpu_j(ph, meter) / ph.duration_s, 2)
        print(f"ready in {E['startup_s']} s, idle {E['idle_w']} W", flush=True)
        for frac in a.fractions:
            reqs = samples[frac]
            key = f"mc:{name}:r{rnd}:f{frac}"
            with meter.phase(key) as ph, Recorder(meter, a.port) as rec:
                out, sent_s, wall = step(a.port, reqs, a, a.offset)
            ok = [x for x in out if "error" not in x]
            prompt = sum(x["prompt"] or 0 for x in ok); cached = sum(x["cached"] or 0 for x in ok)
            gj = gpu_j(ph, meter); outt = sum(x["tokens"] for x in ok)
            s = {"fraction": frac, "requests": len(reqs), "completed": len(ok), "failed": len(out) - len(ok),
                 "unfinished": len(reqs) - len(out), "errors": [x["error"] for x in out if "error" in x][:3],
                 "sent_s": round(sent_s, 1), "wall_s": round(wall, 1), "prompt_tokens": prompt, "cached_tokens": cached,
                 "cache_hit": round(cached / prompt, 4) if prompt else None, "output_tokens": outt,
                 "energy_j": round(gj, 2), "cpu_j": round(ph.energy_j - gj, 2), "avg_w": round(gj / ph.duration_s, 2),
                 "j_per_request": round(gj / len(ok), 2) if ok else None,
                 "j_per_prompt_token": round(gj / prompt, 5) if prompt else None,
                 "j_per_output_token": round(gj / outt, 4) if outt else None,
                 "ttft_p50": pct([x["ttft"] for x in ok], .5), "ttft_p90": pct([x["ttft"] for x in ok], .9),
                 "ttft_p99": pct([x["ttft"] for x in ok], .99), "e2e_p50": pct([x["e2e"] for x in ok], .5),
                 "temp_end_c": gpu_temp(a.gpu), "timeline": rec.timeline(),
                 "rows": {"fields": ["i", "ttft_s", "e2e_s", "prompt", "cached", "out"],
                          "data": [[x["i"], x["ttft"] and round(x["ttft"], 4), round(x["e2e"], 4), x["prompt"], x["cached"], x["tokens"]]
                                   for x in sorted(ok, key=lambda y: y["i"])]}}
            E["steps"].append(s)
            print(f"  frac {frac}: {s['completed']}/{s['requests']} done ({s['unfinished']} unfinished, {s['failed']} failed), "
                  f"hit {s['cache_hit']}, {s['j_per_request']} J/req, {s['j_per_prompt_token']} J/prompt tok, "
                  f"ttft p50/p90/p99 {s['ttft_p50']}/{s['ttft_p90']}/{s['ttft_p99']} s, {s['avg_w']} W", flush=True)
            json.dump(R, open(a.out, "w"), indent=1)
            time.sleep(a.cooldown)
    except Exception as e:  # noqa: BLE001
        E["error"] = repr(e); print(f"!! {name}: {e!r}", flush=True)
    finally:
        stop(proc); time.sleep(10); json.dump(R, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--fractions", default="0.0625,0.125,0.25")
    ap.add_argument("--offset", type=float, default=0, help="start of the replayed window in the trace, seconds")
    ap.add_argument("--window", type=float, default=300)
    ap.add_argument("--drain", type=float, default=300)
    ap.add_argument("--max-output", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--round-start", type=int, default=1, help="first round to run (2 runs only the reversed-order round)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--idle", type=float, default=30)
    ap.add_argument("--settle-window", type=float, default=60)
    ap.add_argument("--cooldown", type=float, default=20)
    ap.add_argument("--startup-timeout", type=float, default=2400)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--predict-only", action="store_true", help="print the sample sizes and LRU hit predictions, then exit")
    ap.add_argument("--out", default="mooncake.json")
    a = ap.parse_args()
    a.fractions = [float(x) for x in a.fractions.split(",")]
    R = [json.loads(l) for l in open(a.trace)]
    keys = group_keys(R)
    samples, info = {}, {}
    for f in a.fractions:
        s, skipped = sample(R, keys, f, a.offset, a.offset + a.window, a.max_len)
        samples[f] = s
        info[str(f)] = {"requests": len(s), "skipped_too_long": skipped, "rate_rps": round(len(s) / a.window, 3),
                        "prompt_tok_per_s": round(sum(r["input_length"] for r in s) / a.window),
                        "lru_hit_pred": {str(c): round(lru_hit(s, c), 4) for c in (244928, 251423, 610351, 10**9)}}
        print(f"fraction {f}: {info[str(f)]}", flush=True)
    if a.predict_only:
        return
    variants = json.load(open(a.variants))
    out = {"trace": os.path.basename(a.trace), "offset_s": a.offset, "window_s": a.window, "drain_s": a.drain,
           "samples": info, "runs": []}
    import joules
    meter = joules.Session(gpus=[a.gpu], cpu=a.cpu, interval=0.1)
    out["joules"], out["meter"] = joules.__version__, meter.devices
    for rnd in range(a.round_start, a.rounds + 1):
        for v in (variants if rnd % 2 else list(reversed(variants))):
            run_variant(v, rnd, a, meter, out, samples)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
