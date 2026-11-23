"""Turn one run of code/agent_ctrl.py (baseline and gate) into the kit's small results file.

    python3 -I gate-kit/summarize.py RAW.json [--out gate-kit-results.json]

For each policy: sessions started and served, turns completed, cache-hit rate, first-turn and resume latency (p50,
p90; both include any wait for admission, as in the paper), and GPU energy per turn when a GPU energy meter was found.
Percentiles follow the paper exactly: resume latency as the benchmark stores it (code/engine_energy2.py: pct, the value
at index int(q n)), first-turn latency as paper/verify_numbers.py recomputes it (index int(q n) - 1). Standard library
only.
"""
import argparse, json, os, sys

KIT = "gate-kit/1"


def pct(v, q):
    """The paper's percentile (code/engine_energy2.py): the value at index int(q x n) of the sorted list."""
    v = sorted(x for x in v if x is not None)
    return round(v[min(len(v) - 1, int(q * len(v)))], 4) if v else None


def p_lo(v, q):
    """paper/verify_numbers.py's percentile for first-turn latency: the value at index int(q x n) - 1."""
    v = sorted(x for x in v if x is not None)
    return round(v[max(0, int(q * len(v)) - 1)], 4) if v else None


def change(a, b):
    return round((b - a) / a * 100, 1) if a and b is not None else None


def policy(E, s, has_gpu_meter):
    f = s["turns_detail"]["fields"]
    rows = [dict(zip(f, r)) for r in s["turns_detail"]["rows"]]
    lat = lambda r: None if r["ttft_s"] is None else r["gate_wait_s"] + r["ttft_s"]
    first = [lat(r) for r in rows if r["turn"] == 0]
    resume = [lat(r) for r in rows if r["turn"] > 0]
    ctrl = s.get("controller") or {}
    trace = ctrl.get("trace") or []
    cap = ctrl.get("capacity")
    out = {
        "name": E["variant"], "policy": E.get("policy"), "engine": E["engine"], "engine_version": E.get("version"),
        "sessions": s["sessions"], "sessions_served": len({r["session"] for r in rows}), "turns": s["turns"],
        "failed_requests": s["failed"], "errors": s.get("errors") or [],
        "cache_hit": s["cache_hit"],
        "first_turn_latency_p50_s": p_lo(first, .5), "first_turn_latency_p90_s": p_lo(first, .9),
        "resume_latency_p50_s": pct(resume, .5), "resume_latency_p90_s": pct(resume, .9),
        "gpu_j_per_turn": s["j_per_turn"] if has_gpu_meter else None,
        "min_turns_per_session": min(collections_count(rows).values()) if rows else 0,
        "step_s": s["wall_s"],
    }
    if ctrl:
        on = [t for t in trace if "gate_on" in t]
        out["gate"] = {"kv_capacity_tokens": cap, "theta": None,
                       "peak_working_set_over_cache": round(max(t["w"] for t in on) / cap, 3) if on and cap else None,
                       "share_of_intervals_gate_on": round(sum(1 for t in on if t["gate_on"]) / len(on), 3) if on else None,
                       "slots_min": min((t["slots"] for t in trace), default=None)}
    return out


def collections_count(rows):
    n = {}
    for r in rows:
        n[r["session"]] = n.get(r["session"], 0) + 1
    return n


def summarize(R):
    devices = R.get("meter") or []
    has_gpu = any(d.get("kind") == "gpu" for d in devices)
    pols = []
    for E in R["runs"]:
        for s in E["steps"]:
            p = policy(E, s, has_gpu)
            v = next((x for x in R.get("variants", []) if x["name"] == E["variant"]), {})
            if p.get("gate"):
                p["gate"]["theta"] = (v.get("admit") or {}).get("theta")
            pols.append(p)
    base = next((p for p in pols if p["policy"] == "gpu-only"), None)
    gate = next((p for p in pols if p.get("gate")), None)
    st = R.get("settings", {})
    out = {
        "kit": KIT,
        "model": R.get("model"), "gpu": R.get("gpu"), "host": R.get("host"),
        "settings": {k: st.get(k) for k in ("sessions", "pauses", "duration", "context", "max_len", "tool_tokens", "answer_tokens", "rounds")},
        "meter": {"joules": R.get("joules"), "devices": devices, "notes": R.get("meter_notes"),
                  "gpu_energy": "measured" if has_gpu else "not measured: no GPU energy meter was found on this machine"},
        "errors": {E["variant"]: E["error"] for E in R["runs"] if E.get("error")},
        "skipped": R.get("skipped") or {},
        "policies": pols,
        "started_at": R.get("started_at"), "finished_at": R.get("finished_at"),
    }
    if base and gate:
        out["gate_vs_baseline"] = {
            "turns_change_pct": change(base["turns"], gate["turns"]),
            "gpu_j_per_turn_change_pct": change(base["gpu_j_per_turn"], gate["gpu_j_per_turn"]),
            "sessions_served": [base["sessions_served"], gate["sessions_served"]],
            "first_turn_latency_p90_s": [base["first_turn_latency_p90_s"], gate["first_turn_latency_p90_s"]],
            "resume_latency_p90_s": [base["resume_latency_p90_s"], gate["resume_latency_p90_s"]],
            "cache_hit": [base["cache_hit"], gate["cache_hit"]],
        }
    return out


def table(S):
    cols = [("policy", "policy"), ("sessions_served", "served"), ("turns", "turns"), ("cache_hit", "cache hit"),
            ("first_turn_latency_p50_s", "first p50 s"), ("first_turn_latency_p90_s", "first p90 s"),
            ("resume_latency_p50_s", "resume p50 s"), ("resume_latency_p90_s", "resume p90 s"), ("gpu_j_per_turn", "GPU J/turn")]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for p in S["policies"]:
        lines.append("| " + " | ".join("—" if p.get(k) is None else str(p.get(k)) for k, _ in cols) + " |")
    g = S.get("gate_vs_baseline")
    if g:
        lines.append("")
        lines.append(f"Gate against baseline: turns {g['turns_change_pct']:+}%, GPU energy per turn "
                     + (f"{g['gpu_j_per_turn_change_pct']:+}%" if g["gpu_j_per_turn_change_pct"] is not None else "not measured")
                     + f", first-turn p90 {g['first_turn_latency_p90_s'][0]} -> {g['first_turn_latency_p90_s'][1]} s"
                     + f", sessions served {g['sessions_served'][0]} -> {g['sessions_served'][1]}.")
    if S["meter"]["gpu_energy"] != "measured":
        lines.append("GPU energy was not measured on this machine (no GPU energy meter found); the other columns stand.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("--out", default=None, help="default: gate-kit-results.json beside RAW")
    a = ap.parse_args()
    S = summarize(json.load(open(a.raw)))
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.raw)), "gate-kit-results.json")
    json.dump(S, open(out, "w"), indent=1)
    print(table(S))
    print(f"\nsaved {out}")
    return 0 if not S["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
