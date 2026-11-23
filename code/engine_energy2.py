"""Engine energy, part 2: energy per output token against latency, default vs tuned engines, realistic open-loop load.

Each variant (an engine plus its flags, from variants.json) is started in turn on the same GPU. For each of two
workloads, requests arrive on a fixed Poisson schedule (the same schedule for every variant) at a sweep of rates;
GPU (and CPU, where the host allows) energy is read with joules over each step, with time to first token, time per
output token and end-to-end latency per request. Variants run --rounds times in alternating order. A quality step
saves greedy answers to 16 fixed prompts so engines can be compared on output, not only speed.

    python engine_energy2.py --model Qwen/Qwen3-32B --variants variants.json --rounds 3 --price 3.7 --out part2.json

Workloads:
  chat  every prompt unique (no shared prefix); prompt and answer lengths log-normal (medians ~800 and ~200 tokens)
  rag   a ~4,000-token document from a pool of 64 (Zipf popularity) plus a unique question; the prefix cache hits
        whenever a popular document is still cached, so the hit rate comes out of the workload instead of being set
"""
import argparse, csv, http.client, io, json, math, os, platform, random, re, shutil, signal, statistics, subprocess, sys, threading, time, urllib.request, zlib

QUESTIONS = ["Explain what this code does and list three bugs you can find.",
             "Write unit tests for the two most important functions above.",
             "Summarize the module in five bullet points.",
             "Refactor the slowest function and explain why it is faster.",
             "Which inputs would make this code fail? Give examples."]
CODE = ("def handler_{i}(event, ctx):\n    total_{i} = sum(x['amount'] for x in event['items'] if x['id'] % {k} == 0)\n"
        "    return {{'id': {i}, 'total': total_{i}, 'region': '{r}'}}\n\n")
QUALITY_PROMPTS = [
    "What is the capital of Australia? Answer in one sentence.", "Write a haiku about GPUs.",
    "Explain prefix caching in LLM serving in two sentences.", "What is 17 * 23? Show the steps.",
    "Write a Python function that reverses a linked list.", "List three causes of the French Revolution.",
    "Translate 'energy per token' into French and German.", "What does a GPU power limit do?",
    "Give a SQL query that finds duplicate emails in a users table.", "Explain batching in LLM inference to a 10 year old.",
    "What is the difference between TCP and UDP?", "Write a regular expression that matches an IPv4 address.",
    "Name the planets of the solar system in order.", "Why is the sky blue? Two sentences.",
    "Write a bash one-liner that counts lines in all .py files.", "What is a mixture-of-experts model?"]


def document(tokens, rng):
    parts, n, i = [], 0, 0
    while n < tokens * 3.2:                      # ~3.2 characters per token for this text; usage gives the real count
        c = CODE.format(i=i, k=rng.randint(2, 97), r=rng.choice(["eu", "us", "ap", "ca"]))
        parts.append(c); n += len(c); i += 1
    return "".join(parts)


def lognormal(rng, median, sigma, lo, hi):
    return int(min(hi, max(lo, round(math.exp(math.log(median) + sigma * rng.gauss(0, 1))))))


def schedule(workload, rate, duration, seed, docs):
    """The requests of one step: (send time, prompt, max_tokens), identical for every variant."""
    rng = random.Random(seed)
    t, out = 0.0, []
    zipf = [1 / (k + 1) ** 1.1 for k in range(len(docs))]
    while True:
        t += rng.expovariate(rate)
        if t > duration:
            return out
        tag = f"{rng.getrandbits(64):x}"
        if workload == "chat":
            body = document(lognormal(rng, 800, 0.8, 50, 6000), rng)
            prompt = f"Session {tag}.\n\n{body}\n{rng.choice(QUESTIONS)}"
            out.append((t, prompt, lognormal(rng, 200, 0.7, 16, 1024)))
        else:
            d = rng.choices(range(len(docs)), weights=zipf)[0]
            prompt = f"Reference document {d}:\n\n{docs[d]}\nQuestion {tag}: {rng.choice(QUESTIONS)}"
            out.append((t, prompt, lognormal(rng, 150, 0.7, 16, 768)))


def ask(port, prompt, max_tokens, ignore_eos=True, temperature=0.0, timeout=900):
    body = json.dumps({"model": "bench", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": temperature, "ignore_eos": ignore_eos, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    t0 = time.perf_counter()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    ttft, usage, text = None, None, []
    try:
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
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
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
    finally:
        conn.close()
    total = time.perf_counter() - t0
    usage = usage or {}
    toks = usage.get("completion_tokens") or 0
    return {"tokens": toks, "prompt": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft": ttft, "e2e": total, "tpot": (total - ttft) / (toks - 1) if ttft is not None and toks > 1 else None,
            "text": "".join(text)}


def run_step(port, reqs, max_inflight=1024):
    """Send each request at its scheduled time (open loop); wait for all of them."""
    res, errs, lock, threads = [], [], threading.Lock(), []
    sem = threading.Semaphore(max_inflight)
    def one(prompt, mt, sent):
        try:
            r = ask(port, prompt, mt)
            r.pop("text", None)
            r["sent"] = sent
            with lock: res.append(r)
        except Exception as e:  # noqa: BLE001
            with lock: errs.append(repr(e)[:300])
        finally:
            sem.release()
    t0 = time.perf_counter()
    for at, prompt, mt in reqs:
        delay = t0 + at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        sem.acquire()
        th = threading.Thread(target=one, args=(prompt, mt, round(time.perf_counter() - t0, 4)), daemon=True)
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    return time.perf_counter() - t0, res, errs


def engine_counters(port):
    """The engine's token, request and cache counters from /metrics (vLLM and SGLang names), summed per name."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not re.search(r"token|request|running|waiting|cache|preempt", line):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$", line.strip())
        if m and not m.group(1).endswith(("_bucket", "_created")):
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


class Recorder:
    """During one step: GPU joules every 0.5 s and the engine's counters every second, as a timeline."""

    def __init__(self, meter, port):
        self.meter, self.port = meter, port
        self.gpus = {d["name"] for d in meter.devices if d["kind"] == "gpu"}
        self.energy, self.counters = [], []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        n = 0
        while not self._stop.is_set():
            t = time.perf_counter() - self.t0
            snap = self.meter.snapshot()
            self.energy.append((round(t, 3), round(sum(j for k, j in snap["devices_j"].items() if k in self.gpus), 3)))
            if n % 2 == 0:
                self.counters.append((round(t, 3), engine_counters(self.port)))
            n += 1
            self._stop.wait(0.5)

    def __enter__(self):
        self.t0 = time.perf_counter(); self._th.start(); return self

    def __exit__(self, *exc):
        self._stop.set(); self._th.join()

    def timeline(self):
        names = sorted({k for _, c in self.counters for k in c})
        return {"energy_gpu_j": self.energy, "counter_names": names,
                "counters": [(t, [c.get(k) for k in names]) for t, c in self.counters]}


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return round(v[min(len(v) - 1, int(q * len(v)))], 4) if v else None


def gpu_temp(gpu):
    try:
        return float(subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def settle(gpu, window=60, tol=1.0, timeout=600):
    t0, hist = time.monotonic(), []
    if gpu_temp(gpu) is None:
        return {"settled": None}
    while True:
        now, c = time.monotonic(), gpu_temp(gpu)
        if c is not None:
            hist.append((now, c))
        recent = [x for t, x in hist if t >= now - window]
        if now - t0 >= window and recent and max(recent) - min(recent) <= tol:
            return {"settled": True, "waited_s": round(now - t0, 1), "temp_c": recent[-1]}
        if now - t0 >= timeout:
            return {"settled": False, "waited_s": round(now - t0, 1), "temp_c": recent[-1] if recent else None}
        time.sleep(1.0)


def gpu_j(ph, meter):
    """The GPU's share of a phase: the headline numbers are GPU board energy, as in part 1."""
    kinds = {d["name"]: d["kind"] for d in meter.devices}
    return sum(j for k, j in (ph.devices or {}).items() if kinds.get(k) == "gpu")


TIMELINES = {}   # step name -> energy and engine-counter timeline (saved beside the results)


MODULE = {"vllm": "vllm.entrypoints.openai.api_server", "sglang": "sglang.launch_server"}


def base_cmd(engine, python, model, port, max_len):
    if engine == "vllm":
        return [python, "-m", MODULE[engine], "--model", model, "--served-model-name", "bench", "--port", str(port),
                "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.90", "--enable-prefix-caching",
                "--enable-prompt-tokens-details"]
    return [python, "-m", MODULE[engine], "--model-path", model, "--served-model-name", "bench", "--port", str(port),
            "--context-length", str(max_len), "--mem-fraction-static", "0.88", "--enable-metrics", "--enable-cache-report"]


def check_flags(v):
    """Every flag a variant adds must appear in its engine's --help; otherwise the variant is skipped, not altered."""
    flags = [f for f in v.get("flags", []) if f.startswith("--")]
    if not flags:
        return []
    try:
        h = subprocess.run([v["python"], "-m", MODULE[v["engine"]], "--help"], capture_output=True, text=True, timeout=300)
        text = h.stdout + h.stderr
    except Exception as e:  # noqa: BLE001
        return [f"could not read --help: {e!r}"]
    return [f for f in flags if not re.search(re.escape(f) + r"(?![\w-])", text)]


def wait_ready(port, proc, timeout, log):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"engine exited with {proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise RuntimeError(f"not ready after {timeout} s; see {log}")


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()


def run_variant(v, rnd, a, meter, R, docs):
    name = v["name"]
    log = os.path.abspath(f"{name}-r{rnd}-server.log")
    cmd = base_cmd(v["engine"], v["python"], a.model, a.port, a.max_len) + v.get("flags", [])
    E = {"variant": name, "engine": v["engine"], "round": rnd, "flags": v.get("flags", []), "command": cmd, "steps": []}
    R["runs"].append(E)
    print(f"\n=== {name} (round {rnd})\n{' '.join(cmd)}", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)),
                                start_new_session=True)
    try:
        E["startup_s"] = round(wait_ready(a.port, proc, a.startup_timeout, log), 1)
        ver = subprocess.run([v["python"], "-c", f"import {v['engine']} as m; print(m.__version__)"], capture_output=True, text=True)
        E["version"] = ver.stdout.strip()
        print(f"ready in {E['startup_s']} s ({v['engine']} {E['version']})", flush=True)
        for w in a.workloads:          # warm-up, unmeasured
            run_step(a.port, schedule(w, 4, 5, 7, docs)[:12])
        E["settle"] = settle(a.gpu, a.settle_window)
        with meter.phase(f"{name}:r{rnd}:idle") as ph:
            time.sleep(a.idle)
        E["idle_w"] = round(gpu_j(ph, meter) / ph.duration_s, 2)        # GPU only, like part 1
        E["idle_devices_w"] = {k: round(j / ph.duration_s, 2) for k, j in (ph.devices or {}).items()}
        print(f"idle {E['idle_w']} W {E['idle_devices_w']}", flush=True)
        for w in a.workloads:
            for rate in a.rates[w]:
                reqs = schedule(w, rate, a.duration, zlib.crc32(f"{w}:{rate}".encode()), docs)
                t0c = gpu_temp(a.gpu)
                with meter.phase(f"{name}:r{rnd}:{w}:{rate}") as ph, Recorder(meter, a.port) as rec:
                    wall, res, errs = run_step(a.port, reqs)
                TIMELINES[f"{name}:r{rnd}:{w}:{rate}"] = rec.timeline()
                out = sum(r["tokens"] for r in res)
                inp = sum(r["prompt"] or 0 for r in res)
                cached = sum(r["cached"] or 0 for r in res)
                g = next(iter((ph.result or {}).get("gpu", {}).values()), {})
                gj = gpu_j(ph, meter)
                s = {"workload": w, "rate": rate, "requests": len(reqs), "completed": len(res), "failed": len(errs),
                     "errors": errs[:3], "input_tokens": inp, "cached_tokens": cached,
                     "cache_hit": round(cached / inp, 4) if inp else None, "output_tokens": out,
                     "wall_s": round(wall, 3), "output_tok_s": round(out / wall, 2) if wall else None,
                     "energy_j": round(gj, 2), "avg_w": round(gj / ph.duration_s, 2),
                     "cpu_j": round(ph.energy_j - gj, 2), "devices_j": {k: round(x, 2) for k, x in (ph.devices or {}).items()},
                     "j_per_output_token": round(gj / out, 4) if out else None,
                     "j_per_request": round(gj / len(res), 2) if res else None,
                     "system_j_per_output_token": round(ph.energy_j / out, 4) if out and ph.energy_j > gj else None,
                     "ttft_p50": pct([r["ttft"] for r in res], .5), "ttft_p90": pct([r["ttft"] for r in res], .9),
                     "ttft_p99": pct([r["ttft"] for r in res], .99),
                     "tpot_p50": pct([r["tpot"] for r in res], .5), "tpot_p90": pct([r["tpot"] for r in res], .9),
                     "e2e_p50": pct([r["e2e"] for r in res], .5), "e2e_p90": pct([r["e2e"] for r in res], .9),
                     "temp_start_c": t0c, "temp_end_c": gpu_temp(a.gpu),
                     "sm_clock_pct": g.get("clock_pct"), "gpu_util_pct": g.get("gpu_util_pct"),
                     "throttled_pct": g.get("throttled_pct"), "throttle_reasons": g.get("throttle_reasons"), "gpu_health": g}
                s["requests_detail"] = {"fields": ["sent_s", "ttft_s", "e2e_s", "output_tokens", "prompt_tokens", "cached_tokens"],
                                        "rows": [[r["sent"], r["ttft"] and round(r["ttft"], 4), round(r["e2e"], 4), r["tokens"],
                                                  r["prompt"], r["cached"]] for r in sorted(res, key=lambda r: r["sent"])]}
                if a.price and out:
                    s["gpu_cost_per_1m_output_tokens"] = round(a.price / 3600 * wall / out * 1e6, 3)
                E["steps"].append(s)
                print(f"  {w:<5} {rate:>5} req/s: {s['completed']}/{s['requests']} ok, {s['output_tok_s']} tok/s, "
                      f"{s['j_per_output_token']} J/tok, ttft p50/p90 {s['ttft_p50']}/{s['ttft_p90']} s, "
                      f"tpot p50 {s['tpot_p50']} s, {s['avg_w']} W, cache {s['cache_hit']}", flush=True)
                json.dump(R, open(a.out, "w"), indent=1)
                json.dump(TIMELINES, open(a.out.replace(".json", "-timeline.json"), "w"))
        q = []
        for p in QUALITY_PROMPTS:      # greedy answers for the output comparison
            try:
                q.append(ask(a.port, p, 96, ignore_eos=False)["text"])
            except Exception as e:  # noqa: BLE001
                q.append(f"ERROR {e!r}")
        E["quality_answers"] = q
    except Exception as e:  # noqa: BLE001
        E["error"] = repr(e)
        print(f"!! {name}: {e!r}", flush=True)
    finally:
        stop(proc)
        time.sleep(10)
        json.dump(R, open(a.out, "w"), indent=1)


def to_csv(R):
    cols = ["variant", "engine", "round", "version", "idle_w", "workload", "rate", "requests", "completed", "failed",
            "input_tokens", "cached_tokens", "cache_hit", "output_tokens", "wall_s", "output_tok_s", "energy_j", "avg_w",
            "j_per_output_token", "j_per_request", "ttft_p50", "ttft_p90", "ttft_p99", "tpot_p50", "tpot_p90", "e2e_p50",
            "e2e_p90", "temp_start_c", "temp_end_c", "sm_clock_pct", "gpu_util_pct", "throttled_pct", "cpu_j",
            "system_j_per_output_token",
            "gpu_cost_per_1m_output_tokens"]
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(cols)
    for E in R["runs"]:
        for s in E["steps"]:
            row = {**s, "variant": E["variant"], "engine": E["engine"], "round": E["round"], "version": E.get("version"),
                   "idle_w": E.get("idle_w")}
            w.writerow([row.get(c, "") for c in cols])
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True, help="JSON list of {name, engine (vllm|sglang), python, flags}")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--workloads", default="chat,rag")
    ap.add_argument("--chat-rates", default="0.5,1,2,4,8,12")
    ap.add_argument("--rag-rates", default="0.5,1,2,4,6,8")
    ap.add_argument("--duration", type=float, default=60, help="seconds of arrivals per step")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--doc-tokens", type=int, default=4000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--idle", type=float, default=30)
    ap.add_argument("--settle-window", type=float, default=60)
    ap.add_argument("--startup-timeout", type=float, default=2400)
    ap.add_argument("--price", type=float, default=0)
    ap.add_argument("--out", default="part2.json")
    ap.add_argument("--cpu", action="store_true", help="also meter the CPU (RAPL), where the host allows it")
    a = ap.parse_args()
    a.workloads = a.workloads.split(",")
    a.rates = {"chat": [float(x) for x in a.chat_rates.split(",")], "rag": [float(x) for x in a.rag_rates.split(",")]}
    variants = json.load(open(a.variants))

    R = {"schema": "engine-energy/2", "model": a.model, "settings": {k: v for k, v in vars(a).items()}, "variants": variants,
         "runs": [], "skipped": {}, "host": {"os": platform.platform(), "python": platform.python_version()},
         "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for v in variants:
        missing = check_flags(v)
        if missing:
            R["skipped"][v["name"]] = f"flags not in this engine's --help: {missing}"
            print(f"!! skipping {v['name']}: flags not found {missing}", flush=True)
    variants = [v for v in variants if v["name"] not in R["skipped"]]

    import joules
    meter = joules.Session(gpus=[a.gpu], cpu=a.cpu, interval=0.1)
    R["joules"], R["meter"], R["meter_notes"] = joules.__version__, meter.devices, meter.notes
    if shutil.which("nvidia-smi"):
        R["gpu"] = subprocess.run(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=name,driver_version,power.limit,clocks.max.sm",
                                   "--format=csv"], capture_output=True, text=True).stdout.strip()
    print(f"meter: {meter.devices}\nvariants: {[v['name'] for v in variants]}", flush=True)
    docs = [document(a.doc_tokens, random.Random(1000 + i)) for i in range(a.docs)]
    for rnd in range(1, a.rounds + 1):
        order = variants if rnd % 2 else variants[::-1]
        for v in order:
            run_variant(v, rnd, a, meter, R, docs)
    R["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meter.close()
    json.dump(R, open(a.out, "w"), indent=1)
    text = to_csv(R)
    open(a.out.replace(".json", ".csv"), "w").write(text)
    json.dump(TIMELINES, open(a.out.replace(".json", "-timeline.json"), "w"))
    for f in (a.out, a.out.replace(".json", ".csv"), a.out.replace(".json", "-timeline.json")):   # also next to Jupyter's root, where its file panel opens
        try:
            shutil.copy(f, os.path.join("/", os.path.basename(f)))
        except OSError:
            pass
    print("\n=== CSV ===\n" + text + f"=== saved {a.out} and its .csv (copies in /)")


if __name__ == "__main__":
    main()
