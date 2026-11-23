"""E3: keep, offload or recompute? GPU (and CPU) energy of KV-cache policies for agent sessions that pause.

Each agent session starts from its own ~8k-token context, then repeats: send the whole history plus a new tool
result, get a 200-token answer, pause (a tool call, log-normal around the median pause), append a fixed summary of
the turn and the tool's output, and go again until the step ends. Prompts are identical for every engine and policy
(the appended text is synthetic, not the model's answer), so only the cache policy changes what has to be recomputed.

Policies are engine variants (variants-agent.json): KV kept in GPU memory only (the default prefix cache), offloaded
to host memory (vLLM KV offloading, SGLang hierarchical cache), or not cached at all. A variant whose flags are not in
its engine's --help is skipped and recorded, never altered.

    python agent_energy.py --model Qwen/Qwen3-32B --variants variants-agent.json --sessions 16,96 --pauses 5,30 \
        --rounds 2 --cpu --price 3.7 --out agents.json
"""
import argparse, json, math, os, platform, random, shutil, subprocess, sys, threading, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_energy2 import (Recorder, ask, base_cmd, check_flags, document, gpu_j, gpu_temp, pct, settle, stop,  # noqa: E402
                            wait_ready)

SUMMARY = "Assistant step {t}: I read the output above, updated the plan and ran the next check on module {m}.\n"


def tool_output(rng, t, tokens):
    lines, n, i = [], 0, 0
    while n < tokens * 3.2:
        s = f"line {i}: status={'ok' if rng.random() > .1 else 'fail'} value={rng.randint(0, 99999)} path=/srv/app/module_{rng.randint(0, 999)}.py\n"
        lines.append(s); n += len(s); i += 1
    return f"Tool output for step {t}:\n" + "".join(lines)


class Gate:
    """Session admission: at most `slots` sessions may send turns; a session keeps its slot through its pauses (so its
    KV stays warm) and gives it up after `turns` turns, then queues again (FIFO). The engine is unchanged."""
    def __init__(self, slots, turns):
        self.slots, self.turns, self.active, self.queue = slots, turns, {}, []
        self.cv = threading.Condition()

    def acquire(self, sid, deadline):
        """Seconds waited for a slot, or None if the step ended first."""
        t0 = time.perf_counter()
        with self.cv:
            if sid in self.active:
                return 0.0
            self.queue.append(sid)
            while not (len(self.active) < self.slots and self.queue[0] == sid):
                left = deadline - time.perf_counter()
                if left <= 0:
                    self.queue.remove(sid); self.cv.notify_all(); return None
                self.cv.wait(min(left, 1.0))
            self.queue.pop(0); self.active[sid] = 0; self.cv.notify_all()
        return time.perf_counter() - t0

    def done_turn(self, sid):
        with self.cv:
            self.active[sid] += 1
            if self.active[sid] >= self.turns:
                del self.active[sid]; self.cv.notify_all()


def session(sid, port, deadline, a, out, lock, pause_median, seed, gate=None):
    """One agent: turns until the deadline; every request is recorded."""
    rng = random.Random(seed)
    history = f"Agent session {sid:04d}. Repository under review:\n\n" + document(a.context, rng) + "\n"
    t = 0
    while time.perf_counter() < deadline:
        history += tool_output(rng, t, a.tool_tokens) + "\nWhat is the next step?\n"
        wait = 0.0
        if gate:
            wait = gate.acquire(sid, deadline)
            if wait is None:
                break
        try:
            r = ask(port, history, a.answer_tokens)
            r.pop("text", None)
            r.update(session=sid, turn=t, gate_wait=wait)
            with lock: out.append(r)
        except Exception as e:  # noqa: BLE001
            with lock: out.append({"session": sid, "turn": t, "error": repr(e)[:200]})
        if gate:
            gate.done_turn(sid)
        history += SUMMARY.format(t=t, m=rng.randint(0, 999))
        t += 1
        pause = math.exp(math.log(pause_median) + 0.8 * rng.gauss(0, 1))
        time.sleep(max(0.0, min(pause, deadline - time.perf_counter(), pause_median * 6)))


def run_policy(v, rnd, a, meter, R, timelines):
    name = v["name"]
    log = os.path.abspath(f"{name}-agent-r{rnd}-server.log")
    cmd = base_cmd(v["engine"], v["python"], a.model, a.port, a.max_len) + v.get("flags", [])
    if "--disable-prefix-caching" in cmd:
        cmd = [c for c in cmd if c != "--enable-prefix-caching"]
    E = {"variant": name, "engine": v["engine"], "policy": v.get("policy"), "round": rnd, "flags": v.get("flags", []),
         "command": cmd, "steps": []}
    R["runs"].append(E)
    print(f"\n=== {name} (round {rnd})\n{' '.join(cmd)}", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)),
                                start_new_session=True)
    try:
        E["startup_s"] = round(wait_ready(a.port, proc, a.startup_timeout, log), 1)
        ver = subprocess.run([v["python"], "-c", f"import {v['engine']} as m; print(m.__version__)"], capture_output=True, text=True)
        E["version"] = ver.stdout.strip()
        for i in range(4):   # warm-up, unmeasured
            ask(a.port, f"Warm-up {i}.\n" + document(2000, random.Random(i)), 16)
        E["settle"] = settle(a.gpu, a.settle_window)
        with meter.phase(f"agent:{name}:r{rnd}:idle") as ph:
            time.sleep(a.idle)
        E["idle_w"] = round(gpu_j(ph, meter) / ph.duration_s, 2)
        print(f"ready in {E['startup_s']} s, idle {E['idle_w']} W", flush=True)
        for S in a.sessions:
            for P in a.pauses:
                out, lock = [], threading.Lock()
                key = f"agent:{name}:r{rnd}:s{S}:p{P}"
                gate = Gate(v["admit"]["slots"], v["admit"]["turns"]) if v.get("admit") else None
                with meter.phase(key) as ph, Recorder(meter, a.port) as rec:
                    t0 = time.perf_counter(); deadline = t0 + a.duration
                    th = [threading.Thread(target=session, args=(i, a.port, deadline, a, out, lock, P,
                                                                 zlib.crc32(f"{S}:{P}:{i}".encode()), gate), daemon=True)
                          for i in range(S)]
                    for i, x in enumerate(th):
                        x.start(); time.sleep(min(0.25, a.duration / 20 / max(S, 1)))   # sessions start staggered
                    for x in th:
                        x.join()
                    wall = time.perf_counter() - t0
                timelines[key] = rec.timeline()
                ok = [r for r in out if "error" not in r]
                gj = gpu_j(ph, meter)
                outt = sum(r["tokens"] for r in ok)
                prompt = sum(r["prompt"] or 0 for r in ok)
                cached = sum(r["cached"] or 0 for r in ok)
                resumed = [r for r in ok if r["turn"] > 0]
                s = {"sessions": S, "pause_median_s": P, "turns": len(ok), "failed": len(out) - len(ok),
                     "errors": [r["error"] for r in out if "error" in r][:3], "wall_s": round(wall, 2),
                     "output_tokens": outt, "prompt_tokens": prompt, "cached_tokens": cached,
                     "recomputed_tokens": prompt - cached, "cache_hit": round(cached / prompt, 4) if prompt else None,
                     "energy_j": round(gj, 2), "cpu_j": round(ph.energy_j - gj, 2), "avg_w": round(gj / ph.duration_s, 2),
                     "j_per_output_token": round(gj / outt, 4) if outt else None,
                     "j_per_turn": round(gj / len(ok), 2) if ok else None,
                     "system_j_per_turn": round(ph.energy_j / len(ok), 2) if ok else None,
                     "resume_ttft_p50": pct([r["ttft"] for r in resumed], .5), "resume_ttft_p90": pct([r["ttft"] for r in resumed], .9),
                     "first_ttft_p50": pct([r["ttft"] for r in ok if r["turn"] == 0], .5),
                     "context_tokens_p50": pct([r["prompt"] for r in ok], .5), "context_tokens_max": max([r["prompt"] or 0 for r in ok] or [0]),
                     "temp_end_c": gpu_temp(a.gpu),
                     "admit_slots": (v.get("admit") or {}).get("slots"), "admit_turns": (v.get("admit") or {}).get("turns"),
                     "gate_wait_p50": pct([r["gate_wait"] for r in resumed], .5),
                     "gate_wait_p90": pct([r["gate_wait"] for r in resumed], .9),
                     "resume_latency_p50": pct([r["gate_wait"] + r["ttft"] for r in resumed if r["ttft"] is not None], .5),
                     "resume_latency_p90": pct([r["gate_wait"] + r["ttft"] for r in resumed if r["ttft"] is not None], .9),
                     "turns_detail": {"fields": ["session", "turn", "ttft_s", "e2e_s", "prompt_tokens", "cached_tokens", "output_tokens",
                                                 "gate_wait_s"],
                                      "rows": [[r["session"], r["turn"], r["ttft"] and round(r["ttft"], 4), round(r["e2e"], 4),
                                                r["prompt"], r["cached"], r["tokens"], round(r["gate_wait"], 4)] for r in ok]}}
                if a.price and outt:
                    s["gpu_cost_per_1m_output_tokens"] = round(a.price / 3600 * wall / outt * 1e6, 3)
                E["steps"].append(s)
                print(f"  sessions {S:>3}, pause {P:>3} s: {s['turns']} turns, {s['failed']} failed, cache hit {s['cache_hit']}, "
                      f"{s['j_per_turn']} J/turn ({s['system_j_per_turn']} with CPU), resume ttft p50/p90 "
                      f"{s['resume_ttft_p50']}/{s['resume_ttft_p90']} s, {s['avg_w']} W"
                      + (f", with gate wait p50/p90 {s['resume_latency_p50']}/{s['resume_latency_p90']} s" if gate else ""), flush=True)
                json.dump(R, open(a.out, "w"), indent=1)
                json.dump(timelines, open(a.out.replace(".json", "-timeline.json"), "w"))
                time.sleep(a.cooldown)   # let the next step start from a quieter card
    except Exception as e:  # noqa: BLE001
        E["error"] = repr(e)
        print(f"!! {name}: {e!r}", flush=True)
    finally:
        stop(proc)
        time.sleep(10)
        json.dump(R, open(a.out, "w"), indent=1)


def to_csv(R):
    cols = ["variant", "engine", "policy", "round", "idle_w", "sessions", "pause_median_s", "turns", "failed", "wall_s",
            "output_tokens", "prompt_tokens", "cached_tokens", "recomputed_tokens", "cache_hit", "energy_j", "cpu_j", "avg_w",
            "j_per_output_token", "j_per_turn", "system_j_per_turn", "resume_ttft_p50", "resume_ttft_p90", "first_ttft_p50",
            "context_tokens_p50", "context_tokens_max", "temp_end_c", "gpu_cost_per_1m_output_tokens",
            "admit_slots", "admit_turns", "gate_wait_p50", "gate_wait_p90", "resume_latency_p50", "resume_latency_p90"]
    lines = [",".join(cols)]
    for E in R["runs"]:
        for s in E["steps"]:
            row = {**s, "variant": E["variant"], "engine": E["engine"], "policy": E.get("policy"), "round": E["round"],
                   "idle_w": E.get("idle_w")}
            lines.append(",".join("" if row.get(c) is None else str(row.get(c)) for c in cols))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--sessions", default="16,96")
    ap.add_argument("--pauses", default="5,30", help="median pause between turns, seconds")
    ap.add_argument("--duration", type=float, default=180, help="seconds per step")
    ap.add_argument("--context", type=int, default=8000, help="approx tokens of each session's starting context")
    ap.add_argument("--tool-tokens", type=int, default=400)
    ap.add_argument("--answer-tokens", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--idle", type=float, default=30)
    ap.add_argument("--settle-window", type=float, default=60)
    ap.add_argument("--cooldown", type=float, default=20)
    ap.add_argument("--startup-timeout", type=float, default=2400)
    ap.add_argument("--price", type=float, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="agents.json")
    a = ap.parse_args()
    a.sessions = [int(x) for x in a.sessions.split(",")]
    a.pauses = [float(x) for x in a.pauses.split(",")]
    variants = json.load(open(a.variants))
    R = {"schema": "agent-energy/1", "model": a.model, "settings": vars(a), "variants": variants, "runs": [], "skipped": {},
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
    R["joules"], R["meter"], R["meter_notes"] = joules.__version__, meter.devices, meter.notes
    if shutil.which("nvidia-smi"):
        R["gpu"] = subprocess.run(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=name,driver_version,power.limit,memory.total",
                                   "--format=csv"], capture_output=True, text=True).stdout.strip()
    print(f"meter: {meter.devices}\npolicies: {[v['name'] for v in variants]}", flush=True)
    timelines = {}
    for rnd in range(1, a.rounds + 1):
        for v in (variants if rnd % 2 else variants[::-1]):
            run_policy(v, rnd, a, meter, R, timelines)
    R["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meter.close()
    json.dump(R, open(a.out, "w"), indent=1)
    text = to_csv(R)
    open(a.out.replace(".json", ".csv"), "w").write(text)
    for f in (a.out, a.out.replace(".json", ".csv"), a.out.replace(".json", "-timeline.json")):
        try:
            shutil.copy(f, os.path.join("/", os.path.basename(f)))
        except OSError:
            pass
    print("\n=== CSV ===\n" + text + f"=== saved {a.out} and its .csv (copies in /)")


if __name__ == "__main__":
    main()
