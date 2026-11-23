#!/usr/bin/env python3
"""Analysis layer for the KV-cache energy study (private).

Regenerates every table, model check, CPU bound and figure from the two result zips:

    python3 -I analyze.py            # from anywhere; paths are relative to this file

Inputs (read only): ../results/all-results.zip, ../results/extra-results.zip.
Each zip is extracted into its own fresh directory (data/main, data/extra); nothing inside them is executed or imported.
Outputs (next to this file): tables.md, models.md, fairness.md, cpu.md, summary.md, figures/*.png.

Python 3 standard library + matplotlib only. All energy is GPU board energy (NVML) unless labeled otherwise.
"""
import bisect
import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import statistics
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
ZIPS = {"main": "all-results.zip", "extra": "extra-results.zip"}

# Constants (theory doc, plan doc)
C_GPU = 244_928              # vLLM GPU KV capacity, tokens (vLLM log)
KV_BYTES_PER_TOKEN = 262_144  # Qwen3-32B BF16
COOLDOWN_S = 20.0            # agent_energy.py --cooldown default (not overridden in run_all.sh / extra.sh)
W_PER_CORE = 10.0            # pessimistic host power per busy core (task spec)
J_PER_BYTE = 1e-9            # ASSUMPTION for a host-memory/PCIe bound: 1 nJ per byte moved (pessimistic; not measured)


def flops_per_token(L):
    """Theory sec. 1: F(L) = 62.4 GFLOP + 2.10 MFLOP x L/2 per prompt token (causal prefill of L tokens)."""
    return 62.4e9 + 2.10e6 * L / 2


# ----------------------------------------------------------------------------------------------------------------------
# Extraction (treat the zips as data: no member is executed; paths are sanitised)
# ----------------------------------------------------------------------------------------------------------------------
def extract():
    for key, name in ZIPS.items():
        dst = os.path.join(DATA, key)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        with zipfile.ZipFile(os.path.join(RESULTS, name)) as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                rel = os.path.normpath(m.filename)
                if rel.startswith("..") or os.path.isabs(rel):
                    raise SystemExit(f"refusing unsafe member {m.filename!r} in {name}")
                out = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(m) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)


def load(rel):
    with open(os.path.join(DATA, rel)) as f:
        return json.load(f)


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def utc(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S")


# ----------------------------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------------------------
def pct(v, q):
    """Same nearest-rank definition as the run scripts (engine_energy2.pct)."""
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def f(v, nd=2, pctsign=False):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,d}"
    s = f"{v:,.{nd}f}"
    return s + ("%" if pctsign else "")


def rel(a, b):
    """(a - b) / b in %."""
    if a is None or b is None or b == 0:
        return None
    return 100.0 * (a - b) / b


def sgn(v, nd=1):
    return "n/a" if v is None else f"{v:+.{nd}f}%"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def turns(step):
    td = step["turns_detail"]
    idx = {k: i for i, k in enumerate(td["fields"])}
    return [{k: r[i] for k, i in idx.items()} for r in td["rows"]]


VARIANT_LABEL = {
    "vllm-gpu": "vLLM GPU-only", "vllm-nocache": "vLLM no cache", "vllm-offload": "vLLM offload",
    "sglang-gpu": "SGLang GPU-only (LRU)", "sglang-nocache": "SGLang no cache", "sglang-offload": "SGLang offload 80 GB",
    "sglang-offload160": "SGLang offload 160 GB", "sglang-lfu": "SGLang LFU", "sglang-slru": "SGLang SLRU",
    "vllm-admit16x4": "vLLM gate 16x4", "vllm-admit16x8": "vLLM gate 16x8",
}


# ----------------------------------------------------------------------------------------------------------------------
# Data model: agent cells keyed (variant, sessions, pause) -> {replicate label: step}
# ----------------------------------------------------------------------------------------------------------------------
def agent_cells(doc, label=None, rename=None):
    cells = {}
    for r in doc["runs"]:
        v = r["variant"] if not rename else rename.get(r["variant"], r["variant"])
        for s in r["steps"]:
            key = (v, s["sessions"], s["pause_median_s"])
            cells.setdefault(key, {})[label or f"r{r['round']}"] = s
    return cells


def cell_mean(cells, key, field):
    c = cells.get(key)
    if not c:
        return None
    return mean([s.get(field) for s in c.values()])


def all_ttft(step, q, first_only=False):
    return pct([t["ttft_s"] for t in turns(step) if (t["turn"] == 0 or not first_only)], q)


AGENT_METRICS = [
    ("J/turn (GPU)", lambda s: s["j_per_turn"], 1),
    ("cache hit %", lambda s: 100 * s["cache_hit"] if s["cache_hit"] is not None else None, 3),
    ("turns", lambda s: s["turns"], 0),
    ("resume TTFT p50 s", lambda s: s["resume_ttft_p50"], 3),
    ("resume TTFT p90 s", lambda s: s["resume_ttft_p90"], 3),
    ("TTFT p90 all turns s", lambda s: all_ttft(s, .9), 3),
    ("first-turn TTFT p50 s", lambda s: s["first_ttft_p50"], 3),
]
GATE_METRICS = AGENT_METRICS + [
    ("resume latency incl. gate wait p90 s (as recorded)", lambda s: s.get("resume_latency_p90"), 3),
]


def rep_table(cells, reps, metrics, keyfmt, keyhdr, order=None):
    """One row per (cell, metric): replicate values, mean, difference of the 2nd replicate vs the 1st in %."""
    hdr = keyhdr + ["metric"] + reps + ["mean", f"{reps[-1]} vs {reps[0]}" if len(reps) > 1 else "diff"]
    rows = []
    for key in (order or sorted(cells)):
        c = cells[key]
        for name, fn, nd in metrics:
            vals = [fn(c[r]) if r in c else None for r in reps]
            m = mean(vals)
            d = rel(vals[-1], vals[0]) if len(reps) > 1 else None
            fmtv = (lambda x: f(int(round(x)) if x is not None else None)) if nd == 0 else (lambda x: f(x, nd))
            rows.append(keyfmt(key) + [name] + [fmtv(v) for v in vals] + [f(m, max(nd, 1)), sgn(d, 2)])
    return md_table(hdr, rows)


def agent_keyfmt(key):
    v, S, P = key
    return [VARIANT_LABEL.get(v, v), S, f"{P:g} s"]


def agent_order(cells, variants):
    return sorted(cells, key=lambda k: (variants.index(k[0]) if k[0] in variants else 99, k[1], k[2]))


# ----------------------------------------------------------------------------------------------------------------------
# 1. Tables
# ----------------------------------------------------------------------------------------------------------------------
def build_tables(D):
    T = ["# Tables: every cell, both rounds\n",
         "Generated by `analyze.py` from the two result zips. Energy is GPU board energy (NVML). `r1`/`r2` are the two "
         "rounds; the last column is round 2 relative to round 1 (%). Hit = cached / prompt tokens over all turns of the "
         "step (first turns included). Resume TTFT = turns after a session's first, as recorded by the run scripts; "
         "'TTFT p90 all turns' and the fairness section are computed here from `turns_detail`. Percentiles use the "
         "scripts' own nearest-rank rule.\n"]

    # E2
    T.append("## E2: long prompts, cache miss vs hit (both engines, ONE round)\n")
    T.append("E2 was run with `rounds=1` (settings in both JSONs), so round 2 is n/a. File names give the `--long-tokens` "
             "target (1,600 / 12,600); the measured prompt medians are ~2,044 and ~16,026 tokens (the plan's ~2k and ~16k).\n")
    for fn in ("e2-len1600.json", "e2-len12600.json"):
        d = D[fn]
        cells = {}
        for e in d["engines"]:
            for r in e["rows"]:
                cells[(e["engine"], r["workload"], r["users"])] = {"r1": r}
        metrics = [("J/output token", lambda r: r["joules_per_token"], 4),
                   ("above-idle J/output token", lambda r: r["above_idle_joules_per_token"], 4),
                   ("cached prompt %", lambda r: 100 * r["cached_prompt_fraction"], 2),
                   ("requests", lambda r: r["requests"], 0),
                   ("prompt tokens (median)", lambda r: r["prompt_tokens_median"], 0),
                   ("TTFT p50 s", lambda r: r["ttft_s_median"], 3), ("TTFT p90 s", lambda r: r["ttft_s_p90"], 3)]
        T.append(f"### {fn}\n")
        T.append(rep_table(cells, ["r1", "r2"], metrics, lambda k: [k[0], k[1], k[2]], ["engine", "workload", "users"]))

    # E3 main
    main = agent_cells(D["agents.json"])
    T.append("\n## E3: agent sessions, main matrix (both engines, 2 rounds)\n")
    T.append("vLLM offload at 80 GB failed to start in both rounds (`Insufficient space in /dev/shm ... 81920 MiB "
             "required, 76800 MiB free`, server logs), so it has no steps here; its 70 GiB rerun is the next table.\n")
    order = agent_order(main, ["vllm-gpu", "vllm-nocache", "sglang-gpu", "sglang-offload", "sglang-nocache"])
    T.append(rep_table(main, ["r1", "r2"], AGENT_METRICS, agent_keyfmt, ["variant", "sessions", "pause"], order))

    # P2
    p2 = {}
    for r in D["part2.json"]["runs"]:
        for s in r["steps"]:
            p2.setdefault((r["variant"], s["workload"], s["rate"]), {})[f"r{r['round']}"] = s
    vo = ["vllm-default", "vllm-tuned", "sglang-default", "sglang-tuned"]
    T.append("\n## P2: default vs tuned, open-loop chat and RAG (2 rounds)\n")
    T.append("Tuned flags: vLLM `--max-num-batched-tokens 16384 --max-num-seqs 256`; SGLang `--chunked-prefill-size 16384 "
             "--max-running-requests 256 --schedule-policy lpm`. 'turns' = completed requests.\n")
    p2m = [("J/output token", lambda s: s["j_per_output_token"], 4),
           ("cache hit %", lambda s: 100 * (s["cache_hit"] or 0), 2),
           ("completed requests", lambda s: s["completed"], 0),
           ("TTFT p50 s", lambda s: s["ttft_p50"], 3), ("TTFT p90 s", lambda s: s["ttft_p90"], 3)]
    T.append(rep_table(p2, ["r1", "r2"], p2m, lambda k: [k[0], k[1], f"{k[2]:g}/s"], ["variant", "workload", "rate"],
                       sorted(p2, key=lambda k: (vo.index(k[0]), k[1], k[2]))))

    # Offload 70 GiB
    off = agent_cells(D["agents-offload.json"], rename={"vllm-offload": "vllm-offload70"})
    VARIANT_LABEL["vllm-offload70"] = "vLLM offload 70 GiB"
    T.append("\n## Offload 70 GiB (vLLM native CPU offload, 2 rounds)\n")
    T.append(rep_table(off, ["r1", "r2"], AGENT_METRICS, agent_keyfmt, ["variant", "sessions", "pause"]))

    # Gate
    gate = agent_cells(D["agents-admit.json"])
    T.append("\n## Gate: session admission, 16 slots, yield after 4 or 8 turns (vLLM, 2 rounds)\n")
    T.append("Recorded resume TTFT / latency covers resumed turns only; see the fairness section for waits including "
             "first turns and sessions never served.\n")
    T.append(rep_table(gate, ["r1", "r2"], GATE_METRICS, agent_keyfmt, ["variant", "sessions", "pause"],
                       agent_order(gate, ["vllm-admit16x4", "vllm-admit16x8"])))

    # LFU / SLRU
    ev = agent_cells(D["agents-evict.json"])
    T.append("\n## LFU / SLRU eviction (SGLang GPU-only, 2 rounds)\n")
    T.append(rep_table(ev, ["r1", "r2"], AGENT_METRICS, agent_keyfmt, ["variant", "sessions", "pause"],
                       agent_order(ev, ["sglang-lfu", "sglang-slru"])))

    # Cliff
    cl = agent_cells(D["agents-cliff.json"])
    T.append("\n## Cliff 24-40 sessions (vLLM GPU-only, 2 rounds)\n")
    T.append(rep_table(cl, ["r1", "r2"], AGENT_METRICS, agent_keyfmt, ["variant", "sessions", "pause"]))

    # 160 GB
    h = agent_cells(D["agents-host160.json"])
    T.append("\n## 160 GB host tier (SGLang hierarchical cache, 2 rounds)\n")
    T.append(rep_table(h, ["r1", "r2"], AGENT_METRICS, agent_keyfmt, ["variant", "sessions", "pause"]))

    # Transfer
    tr = {}
    for r in D["transfer.json"]["runs"]:
        for s in r["steps"]:
            for ph in ("recompute", "gpu_hit", "after_flush"):
                tr.setdefault((r["variant"], s["length"], ph), {})[f"rep{s['rep']}"] = s[ph]
    T.append("\n## Transfer vs recompute (SGLang, 8 prompts at once, 1 output token, 3 repetitions)\n")
    T.append("Replicates here are the 3 repetitions (`rep1..rep3`); last column = rep3 vs rep1. `after_flush` = third "
             "send after ~325k filler tokens pushed the prompts out of GPU memory: a host-tier restore for "
             "sglang-host160, a recompute for the GPU-only control. `length` is the script's nominal target; actual "
             "tokens per prompt are in the 'prompt tokens / request' rows (about 1.24-1.27x the nominal).\n")
    trm = [("above-idle J/prompt token", lambda x: x["above_idle_j_per_prompt_token"], 5),
           ("gross J/prompt token", lambda x: x["j_per_prompt_token"], 5),
           ("cached %", lambda x: 100 * x["cached_frac"], 2),
           ("prompt tokens / request", lambda x: x["prompt_tokens"] / x["requests"], 0),
           ("TTFT p50 s", lambda x: x["ttft_p50"], 3)]
    T.append(rep_table(tr, ["rep1", "rep2", "rep3"], trm, lambda k: [k[0], k[1], k[2]], ["variant", "nominal length", "send"],
                       sorted(tr, key=lambda k: (k[0], k[1], ["recompute", "gpu_hit", "after_flush"].index(k[2])))))

    # Seeds
    sd = {}
    for lab in ("s1", "s2"):
        for k, v in agent_cells(D[f"agents-seed-{lab}.json"], label=lab).items():
            sd.setdefault(k, {}).update(v)
    T.append("\n## New seeds s1 / s2 (vLLM GPU-only vs no cache; ONE round per seed)\n")
    T.append("Each seed ran one round, so the two replicate columns are the two seeds (different random sessions), "
             "and the last column is the seed-to-seed difference, not round noise. The original-seed mean (main "
             "matrix) is in the E3 table above.\n")
    T.append(rep_table(sd, ["s1", "s2"], AGENT_METRICS, agent_keyfmt, ["variant", "sessions", "pause"],
                       agent_order(sd, ["vllm-gpu", "vllm-nocache"])))
    return "\n".join(T), {"main": main, "off": off, "gate": gate, "ev": ev, "cl": cl, "h": h, "tr": tr, "sd": sd, "p2": p2}


# ----------------------------------------------------------------------------------------------------------------------
# 2. Fairness for gated runs
# ----------------------------------------------------------------------------------------------------------------------
def fairness(D, C):
    doc = D["agents-admit.json"]
    dur = doc["settings"]["duration"]
    rows, res = [], {}
    for r in doc["runs"]:
        q = next(v["admit"]["turns"] for v in doc["variants"] if v["name"] == r["variant"])
        for s in r["steps"]:
            S = s["sessions"]
            tt = turns(s)
            per = {}
            for t in tt:
                per.setdefault(t["session"], []).append(t)
            served = len(per)
            never = sorted(set(range(S)) - set(per))
            stagger = min(0.25, dur / 20 / max(S, 1))      # agent_admit.py: session i starts i*stagger after t0
            cens = [dur - i * stagger for i in never]       # waited from its start to the deadline (lower bound)
            waits = [t["gate_wait_s"] for t in tt]
            first_w = [t["gate_wait_s"] for t in tt if t["turn"] == 0]
            lat = [t["gate_wait_s"] + t["ttft_s"] for t in tt if t["ttft_s"] is not None]
            readmitted = sum(1 for v in per.values() if len(v) > q)
            yielded_end = sum(1 for v in per.values() if len(v) % q == 0)   # gave the slot back; any re-wait is unrecorded
            maxturns = max(len(v) for v in per.values())
            row = dict(variant=r["variant"], round=r["round"], S=S, P=s["pause_median_s"], q=q, served=served,
                       never=len(never), w50=pct(waits, .5), w90=pct(waits, .9), wmax=max(waits),
                       fw50=pct(first_w, .5), fw90=pct(first_w, .9),
                       wc50=pct(waits + cens, .5), wc90=pct(waits + cens, .9),
                       lat90=pct(lat, .9), lat90c=pct(lat + cens, .9),
                       rec_w90=s.get("gate_wait_p90"), rec_l90=s.get("resume_latency_p90"),
                       readmitted=readmitted, yielded_end=yielded_end, maxturns=maxturns,
                       j=s["j_per_turn"], turns=s["turns"], hit=s["cache_hit"])
            rows.append(row)
            res.setdefault((r["variant"], S, s["pause_median_s"]), []).append(row)
    # GPU-only reference (vLLM main matrix): everyone is served; latency to first token = TTFT
    ref = {}
    for key in [("vllm-gpu", S, P) for S in (48, 96) for P in (5.0, 30.0)]:
        for lab, s in C["main"][key].items():
            tt = turns(s)
            ref.setdefault(key[1:], []).append(dict(served=len({t["session"] for t in tt}), S=key[1],
                                                    lat90=pct([t["ttft_s"] for t in tt], .9),
                                                    flat90=pct([t["ttft_s"] for t in tt if t["turn"] == 0], .9),
                                                    j=s["j_per_turn"], turns=s["turns"]))
    hdr = ["variant", "round", "sessions", "pause", "served / total", "never served", "wait p50 / p90 / max s (all turns)",
           "first-turn wait p50 / p90 s", "wait p50 / p90 incl. censored s", "latency to 1st token p90 s (wait+TTFT, all turns; incl. censored)",
           "recorded gate wait p90 / resume latency p90 s", "sessions re-admitted", "sessions waiting again at end (unrecorded)", "max turns / session"]
    out = []
    for x in sorted(rows, key=lambda x: (x["variant"], x["S"], x["P"], x["round"])):
        out.append([VARIANT_LABEL[x["variant"]], x["round"], x["S"], f"{x['P']:g} s", f"{x['served']} / {x['S']}", x["never"],
                    f"{f(x['w50'],1)} / {f(x['w90'],1)} / {f(x['wmax'],1)}", f"{f(x['fw50'],1)} / {f(x['fw90'],1)}",
                    f"{f(x['wc50'],1)} / {f(x['wc90'],1)}", f"{f(x['lat90'],1)}; {f(x['lat90c'],1)}",
                    f"{f(x['rec_w90'],2)} / {f(x['rec_l90'],2)}", x["readmitted"], x["yielded_end"], x["maxturns"]])
    txt = ["## 2. Fairness of the admission gate (agents-admit.json)\n",
           "Per step: sessions with at least one turn, waits for a slot over **all** turns (first turns included), and "
           "sessions never served. A never-served session's wait is censored at the step end: it waited from its "
           f"staggered start (i x min(0.25, {dur:g}/20/S) s, as in agent_admit.py) to the {dur:g} s deadline; the "
           "'incl. censored' quantiles add those censored values, so they are lower bounds on the true quantiles. "
           "A session that used its q turns and gave its slot back may have been queued again at the end of the step; that "
           "second wait is censored and not recorded anywhere, so the count is shown.\n",
           md_table(hdr, out),
           "\nGPU-only reference (vLLM, main matrix): every session is served; latency to first token = TTFT.\n",
           md_table(["sessions", "pause", "round", "served / total", "TTFT p90 all turns s", "first-turn TTFT p90 s", "J/turn", "turns"],
                    [[k[0], f"{k[1]:g} s", i + 1, f"{x['served']} / {x['S']}", f(x["lat90"], 1), f(x["flat90"], 1), f(x["j"], 1), x["turns"]]
                     for k, v in sorted(ref.items()) for i, x in enumerate(v)])]
    return "\n".join(txt), res, ref


# ----------------------------------------------------------------------------------------------------------------------
# 3. Model checks
# ----------------------------------------------------------------------------------------------------------------------
def models(D, C, caps):
    out = ["# Model checks\n", "Every residual is reported as (predicted - measured) / measured. Cell values are the "
           "mean of the two rounds.\n"]
    R = {}

    # (a) offload energy model
    m, h, off = C["main"], C["h"], C["off"]
    E_t = cell_mean(m, ("sglang-gpu", 48, 5.0), "j_per_turn")
    h_t = cell_mean(m, ("sglang-gpu", 48, 5.0), "cache_hit")
    E_fit = cell_mean(h, ("sglang-offload160", 48, 5.0), "j_per_turn")
    h_fit = cell_mean(h, ("sglang-offload160", 48, 5.0), "cache_hit")
    fpar = (1 - E_fit / E_t) / (h_fit - h_t)
    R["f"] = fpar
    cases = [("SGLang 80 GB", m, "sglang-offload", "sglang-gpu", m, 48, 5.0),
             ("SGLang 80 GB", m, "sglang-offload", "sglang-gpu", m, 48, 30.0),
             ("SGLang 80 GB", m, "sglang-offload", "sglang-gpu", m, 96, 5.0),
             ("SGLang 80 GB", m, "sglang-offload", "sglang-gpu", m, 96, 30.0),
             ("vLLM 70 GiB", off, "vllm-offload70", "vllm-gpu", m, 48, 5.0),
             ("vLLM 70 GiB", off, "vllm-offload70", "vllm-gpu", m, 48, 30.0),
             ("vLLM 70 GiB", off, "vllm-offload70", "vllm-gpu", m, 96, 5.0),
             ("vLLM 70 GiB", off, "vllm-offload70", "vllm-gpu", m, 96, 30.0),
             ("SGLang 160 GB", h, "sglang-offload160", "sglang-gpu", m, 48, 30.0),
             ("SGLang 160 GB", h, "sglang-offload160", "sglang-gpu", m, 96, 5.0),
             ("SGLang 160 GB", h, "sglang-offload160", "sglang-gpu", m, 96, 30.0)]
    rows, ra = [], []
    for lab, cc, v, base, bc, S, P in cases:
        Et = cell_mean(bc, (base, S, P), "j_per_turn"); ht = cell_mean(bc, (base, S, P), "cache_hit")
        Em = cell_mean(cc, (v, S, P), "j_per_turn"); hm = cell_mean(cc, (v, S, P), "cache_hit")
        dh = hm - ht
        pred = Et * (1 - fpar * dh)
        e = rel(pred, Em)
        # the saving the model claims vs the measured saving (a sharper test when dh is small)
        sav_p, sav_m = Et - pred, Et - Em
        ra.append(dict(lab=lab, S=S, P=P, Et=Et, dh=dh, pred=pred, meas=Em, err=e, sav_p=sav_p, sav_m=sav_m))
        rows.append([lab, S, f"{P:g} s", f(Et, 1), f(100 * dh, 2), f(pred, 1), f(Em, 1), sgn(e, 1), f(sav_p, 1), f(sav_m, 1),
                     sgn(rel(sav_p, sav_m), 0) if abs(sav_m) > 5 else "saving < 5 J"])
    R["offload"] = ra
    out += ["## (a) Offload energy: E = E_thrash (1 - f dh)\n",
            f"E_thrash = GPU-only J/turn of the same engine in the same cell; dh = hit(offload) - hit(GPU-only), token-weighted. "
            f"f fitted on SGLang 160 GB, 48 sessions, pause 5 s: E_thrash {E_t:.1f} J, hit {100*h_t:.2f}% -> {100*h_fit:.2f}%, "
            f"measured {E_fit:.1f} J  =>  **f = {fpar:.3f}** (decode/idle share {E_t*(1-fpar):.0f} J of the turn).\n",
            md_table(["tier", "sessions", "pause", "E_thrash J", "dh (pp)", "predicted J", "measured J", "error",
                      "predicted saving J", "measured saving J", "saving error"], rows),
            "\nThe whole-turn error looks small whenever dh is small, because the prediction then stays near E_thrash; the "
            "saving columns show how well the model gets the part it is about.\n"]

    # (b) admission model
    g = C["gate"]
    rows, rb = [], []
    for v, q in (("vllm-admit16x4", 4), ("vllm-admit16x8", 8)):
        for S in (48, 96):
            for P in (5.0, 30.0):
                Em = cell_mean(g, (v, S, P), "j_per_turn")
                Ec_m = cell_mean(m, ("vllm-gpu", S, P), "j_per_turn")
                Ew_m = cell_mean(m, ("vllm-gpu", 16, P), "j_per_turn")
                p_plan = (1030 + (q - 1) * 510) / q
                p_th = (1030 + (q - 1) * 400) / q
                p_meas = (Ec_m + (q - 1) * Ew_m) / q
                # hit check: warm turns hit like resumed turns at 16 sessions (same pause), cold turns ~0
                hw = resumed_hit(m[("vllm-gpu", 16, P)]["r1"])
                hp = (q - 1) / q * hw
                hm = cell_mean(g, (v, S, P), "cache_hit")
                rb.append(dict(v=v, q=q, S=S, P=P, meas=Em, p_plan=p_plan, p_th=p_th, p_meas=p_meas, Ec=Ec_m, Ew=Ew_m, hp=hp, hm=hm))
                rows.append([VARIANT_LABEL[v], S, f"{P:g} s", f(Em, 1), f"{p_plan:.0f} ({sgn(rel(p_plan, Em))})",
                             f"{p_th:.0f} ({sgn(rel(p_th, Em))})",
                             f"{p_meas:.0f} ({sgn(rel(p_meas, Em))}) [E_cold {Ec_m:.0f}, E_warm {Ew_m:.0f}]",
                             f"{100*hp:.1f}", f"{100*hm:.1f}"])
    R["gate"] = rb
    out += ["\n## (b) Admission gate: E(q) = [E_cold + (q-1) E_warm] / q\n",
            "Three parameter sets: the plan's interim constants (E_cold 1,030, E_warm 510 J), the theory doc's (1,030 / 400 J), "
            "and measured per pause: E_cold = vLLM GPU-only at the same sessions and pause (thrashing, hit ~0), E_warm = vLLM "
            "GPU-only at 16 sessions and the same pause (no pressure). Hit check: (q-1)/q x the resumed-turn hit at 16 sessions.\n",
            md_table(["gate", "sessions", "pause", "measured J/turn", "plan constants (err)", "theory constants (err)",
                      "measured E_cold/E_warm (err)", "predicted hit %", "measured hit %"], rows)]

    # (c) recompute J/token vs FLOP model
    tr = D["transfer.json"]
    pts = {}
    for r in tr["runs"]:
        for s in r["steps"]:
            for ph in ("recompute",) + (("after_flush",) if r["variant"] == "sglang-gpu-control" else ()):
                x = s[ph]
                pts.setdefault(s["length"], []).append((x["prompt_tokens"] / x["requests"], x["above_idle_j_per_prompt_token"]))
    agg = {L: (mean([a for a, _ in v]), mean([b for _, b in v]), min(b for _, b in v), max(b for _, b in v), len(v))
           for L, v in pts.items()}
    L8, e8 = agg[8000][0], agg[8000][1]
    k_act = e8 / flops_per_token(L8)
    k_nom = e8 / flops_per_token(8000)
    rows, rc = [], []
    for L in sorted(agg):
        La, e, lo, hi, n = agg[L]
        pa = k_act * flops_per_token(La); pn = k_nom * flops_per_token(L)
        rc.append(dict(L=L, La=La, e=e, pa=pa, pn=pn, ea=rel(pa, e), en=rel(pn, e)))
        rows.append([f"{L:,}", f"{La:,.0f}", f"{e:.5f} ({lo:.5f}-{hi:.5f}, n={n})", f"{pa:.5f}", sgn(rel(pa, e)), f"{pn:.5f}", sgn(rel(pn, e))])
    R["flop"] = dict(rows=rc, k_act=k_act, k_nom=k_nom)
    # E2: marginal above-idle energy of re-reading a prompt (miss - hit) per recomputed token, same engine/users
    e2rows, e2r = [], []
    for fn in ("e2-len1600.json", "e2-len12600.json"):
        for e in D[fn]["engines"]:
            by = {(r["workload"], r["users"]): r for r in e["rows"]}
            for u in (1, 4, 16, 64):
                mi, hi_ = by[("long_miss", u)], by[("long_hit", u)]
                Ea = lambda r: r["above_idle_joules_per_token"] * r["output_tokens"]
                ntok = mi["requests"] * mi["prompt_tokens_median"] * (hi_["cached_prompt_fraction"] - mi["cached_prompt_fraction"])
                marg = (Ea(mi) - Ea(hi_)) / ntok
                pred = k_act * flops_per_token(mi["prompt_tokens_median"])
                e2r.append(dict(fn=fn, eng=e["engine"], u=u, L=mi["prompt_tokens_median"], marg=marg, pred=pred))
                e2rows.append([e["engine"], f"{mi['prompt_tokens_median']:,.0f}", u, f"{marg:.4f}", f"{pred:.4f}", sgn(rel(pred, marg))])
    R["e2marg"] = e2r
    out += ["\n## (c) Recompute J/token vs the FLOP model F(L) = 62.4e9 + 2.10e6 L/2\n",
            f"Above-idle GPU J per prompt token of a full recompute (transfer.json: both variants' `recompute` sends plus the "
            f"GPU-only control's `after_flush`, 3 repetitions). Constant J/FLOP fitted on the 8k point: "
            f"{k_act*1e12:.3f} pJ/FLOP using the actual mean prompt length ({L8:,.0f} tokens), {k_nom*1e12:.3f} pJ/FLOP using "
            f"the nominal 8,000 (the theory doc's estimate is 0.85 pJ/FLOP above idle).\n",
            md_table(["nominal L", "actual tokens / prompt", "measured J/token (range)", "model, actual L", "error",
                      "model, nominal L", "error"], rows),
            "\nCross-check on E2 (different run: vLLM/SGLang with 256-token decodes interleaved): marginal above-idle energy of a "
            "miss over a hit per recomputed prompt token, against the same J/FLOP. At 1-4 users the difference is within the "
            "energy noise of a ~20 s run and is not a usable estimate.\n",
            md_table(["engine", "prompt tokens", "users", "measured marginal J/token", "model J/token", "error"], e2rows)]

    # (d) cliff: hit vs W/C
    cl, rows, rd = C["cl"], [], []
    sources = [("vllm-gpu", m, (16, 48, 96), C_GPU), ("vllm-gpu", cl, (24, 32, 40), C_GPU)]
    for v, cc, Ss, cap in sources:
        for S in Ss:
            for P in (5.0, 30.0):
                for lab, s in sorted(cc[(v, S, P)].items()):
                    rd.append(cliff_point(s, cap, f"vLLM {lab}"))
    sg_cap = caps.get("sglang_gpu") or 251_423
    for S in (16, 48, 96):
        for P in (5.0, 30.0):
            for lab, s in sorted(m[("sglang-gpu", S, P)].items()):
                rd.append(cliff_point(s, sg_cap, f"SGLang {lab}"))
    sat = {P: mean([x["rhit_turn"] for x in rd if x["S"] == 16 and x["P"] == P and x["src"].startswith("vLLM")]) for P in (5.0, 30.0)}
    for x in rd:
        N = x["S"]
        thA, thB = 1.0, N / (1 + 0.632 * (N - 1))
        x["thB"] = thB
        x["predA"] = sat[x["P"]] if x["wc"] <= thA else 0.0
        x["predB"] = sat[x["P"]] if x["wc"] <= thB else 0.0
        rows.append([x["src"], N, f"{x['P']:g} s", f"{x['W']:,}", f"{x['wc']:.2f}", f"{100*x['hit']:.2f}", f"{100*x['rhit_tok']:.2f}",
                     f"{100*x['rhit_turn']:.1f}", f"{100*x['predA']:.0f} ({100*(x['predA']-x['rhit_turn']):+.0f})",
                     f"{thB:.2f}", f"{100*x['predB']:.0f} ({100*(x['predB']-x['rhit_turn']):+.0f})", f(x["j"], 1)])
    # empirical collapse point: W/C where the resumed-turn hit rate crosses half its 16-session value (vLLM)
    cross = {}
    for P in (5.0, 30.0):
        pts_ = sorted({(round(x["wc"], 4), x["rhit_turn"]) for x in rd if x["P"] == P and x["src"].startswith("vLLM")})
        half = sat[P] / 2
        c = None
        for (a, ya), (b, yb) in zip(pts_, pts_[1:]):
            if ya >= half > yb:
                c = a + (ya - half) / (ya - yb) * (b - a)
        last_hit = max((a for a, y in pts_ if y > 0.02), default=None)
        first_zero = min((a for a, y in pts_ if y <= 0.02 and a > (last_hit or 0)), default=None)
        cross[P] = (c, last_hit, first_zero)
    R["cliff"] = dict(points=rd, sat=sat, cross=cross)
    out += ["\n## (d) Cliff: hit vs working set / capacity\n",
            f"W = sum over sessions of each session's largest prompt in the step; C = {C_GPU:,} tokens (vLLM log) for vLLM and "
            f"{sg_cap:,} for SGLang (`max_total_num_tokens` in its log). Hit columns: step hit (token-weighted, all turns), "
            "resumed-turn hit by tokens, and the share of resumed turns that hit (>= 50% of the prompt cached). Model A "
            "(fixed cycle): resumed turns hit iff W/C <= 1. Model B (random order): iff W/C <= N/(1 + 0.632(N-1)) (-> 1.58). "
            "Both are step functions at the 16-session hit level "
            f"({100*sat[5.0]:.1f}% at 5 s, {100*sat[30.0]:.1f}% at 30 s); residuals in percentage points in brackets.\n",
            md_table(["engine / round", "sessions", "pause", "W tokens", "W/C", "step hit %", "resumed hit % (tokens)",
                      "resumed turns that hit %", "model A (resid.)", "B threshold", "model B (resid.)", "J/turn"], rows),
            "\nEmpirical collapse (vLLM, resumed turns that hit falling below half the 16-session level, linear interpolation in W/C): "
            + "; ".join(f"pause {P:g} s: W/C {f(c[0], 2)} (last point with hits > 2%: W/C {f(c[1], 2)}, first point at ~0 after it: {f(c[2], 2)})"
                        for P, c in cross.items()) + ".\n"]
    return "\n".join(out), R


def resumed_hit(step):
    tt = [t for t in turns(step) if t["turn"] > 0]
    return sum(1 for t in tt if t["cached_tokens"] >= 0.5 * t["prompt_tokens"]) / len(tt) if tt else None


def cliff_point(s, cap, src):
    tt = turns(s)
    mx = {}
    for t in tt:
        mx[t["session"]] = max(mx.get(t["session"], 0), t["prompt_tokens"])
    W = sum(mx.values())
    res = [t for t in tt if t["turn"] > 0]
    return dict(src=src, S=s["sessions"], P=s["pause_median_s"], W=W, wc=W / cap, hit=s["cache_hit"],
                rhit_tok=sum(t["cached_tokens"] for t in res) / max(1, sum(t["prompt_tokens"] for t in res)),
                rhit_turn=resumed_hit(s), j=s["j_per_turn"], turns=s["turns"])


# ----------------------------------------------------------------------------------------------------------------------
# 4. CPU-time bound for offload (H5)
# ----------------------------------------------------------------------------------------------------------------------
def load_cpu(rel_):
    T, Cc, M = [], [], []
    with open(os.path.join(DATA, rel_)) as fh:
        for r in csv.DictReader(fh):
            T.append(float(r["unix_s"])); Cc.append(float(r["engine_cpu_s"]))
            M.append(float(r["agents_json_mtime"]) if r["agents_json_mtime"] else None)
    return T, Cc, M


def interp(T, Cc, t):
    if t < T[0] or t > T[-1]:
        return None
    i = bisect.bisect_left(T, t)
    if T[i] == t or i == 0:
        return Cc[i]
    a, b = T[i - 1], T[i]
    return Cc[i - 1] + (Cc[i] - Cc[i - 1]) * (t - a) / (b - a)


def window_cpu(T, Cc, t0, t1):
    """Engine CPU seconds in [t0, t1] as the sum of positive increments between samples (a short-lived engine
    subprocess that exits makes the cumulative sum drop by its total; that is not negative CPU use). Edge samples
    are interpolated. None (with reason) if outside the log or the engine restarts inside (sum falls by > 50%)."""
    if t0 < T[0]:
        return None, f"window starts {T[0]-t0:.0f} s before the CPU logger started"
    if t1 > T[-1]:
        return None, "window ends after the log"
    i0, i1 = bisect.bisect_left(T, t0), bisect.bisect_right(T, t1) - 1
    tot = 0.0
    for i in range(max(i0, 1), min(i1 + 2, len(T))):
        if Cc[i - 1] > 50 and Cc[i] < 0.5 * Cc[i - 1]:
            return None, "engine restarts inside the window"
        inc = max(0.0, Cc[i] - Cc[i - 1])
        a, b = T[i - 1], T[i]
        lo, hi = max(a, t0), min(b, t1)
        if hi > lo:
            tot += inc * (hi - lo) / (b - a)
    return tot, None


def drops_in(T, Cc, t0, t1):
    return sum(1 for i in range(1, len(T)) if t0 <= T[i] <= t1 and Cc[i] < Cc[i - 1] - 0.5 and not (Cc[i - 1] > 50 and Cc[i] < 0.5 * Cc[i - 1]))


def save_events(M, merge_s=5.0):
    """Distinct mtimes of the results JSON, in order. A write of a large JSON can be seen mid-write (two mtimes
    milliseconds apart), so changes within merge_s seconds are one save; its time is the last mtime seen."""
    ev, last = [], None
    for m in M:
        if m is None or m == last:
            continue
        last = m
        if ev and m - ev[-1]["t"] < merge_s:
            ev[-1]["t"] = m; ev[-1]["n"] += 1
        else:
            ev.append({"t": m, "n": 1})
    return ev


def expected_saves(doc, with_final):
    X = []
    for ri, r in enumerate(doc["runs"]):
        for k, s in enumerate(r["steps"]):
            X.append(("step", ri, k))
        X.append(("end", ri, None))
    if with_final:
        X.append(("final", None, None))
    return X


def align(doc, ev):
    """Match observed saves to expected saves from the end backwards (the logger may start late, never ends early here).
    Returns (pairs, violations). Gap rules: consecutive step saves of one run are wall_s + 20 s cooldown + a small
    overhead apart; an end-of-variant save comes 25-120 s after the run's last step (cooldown, engine stop, 10 s)."""
    best = None
    for wf in (False, True):
        X = expected_saves(doc, wf)
        n = min(len(X), len(ev))
        pairs = list(zip(X[-n:], ev[-n:]))
        viol = 0
        for (x0, e0), (x1, e1) in zip(pairs, pairs[1:]):
            gap = e1["t"] - e0["t"]
            if x1[0] == "step":
                w = doc["runs"][x1[1]]["steps"][x1[2]]["wall_s"]
                lo, hi = (w + COOLDOWN_S - 2, w + COOLDOWN_S + 30) if (x0[0] == "step" and x0[1] == x1[1]) else (w + COOLDOWN_S, 1e9)
                viol += not (lo <= gap <= hi)
            elif x1[0] == "end":
                viol += not ((25 <= gap <= 120) if x0[0] == "step" else (0 <= gap <= 4000))
            else:
                viol += not (0 <= gap <= 120)
        cand = (viol, -n, wf, pairs)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return best[3], best[0], best[2]


def stops(T, Cc, t0, t1, minc=200.0):
    """Engine stop times: rows where cumulative engine CPU falls below half of the previous row's (> minc s)."""
    return [T[i] for i in range(1, len(T)) if t0 <= T[i] <= t1 and Cc[i - 1] > minc and Cc[i] < 0.5 * Cc[i - 1]]


def cpu_section(D, C):
    out = ["## 4. CPU-time bound for offload (H5)\n"]
    R = {"steps": [], "notes": []}
    Tm, Cm, Mm = load_cpu("extra/cpu-time.csv")
    To, Co, Mo = load_cpu("extra/cpu-time-offload.csv")

    def do_aligned(doc, T, Cc, M, tag):
        ev = save_events(M)
        pairs, viol, wf = align(doc, ev)
        res, ovh = [], []
        merged = sum(1 for e in ev if e["n"] > 1)
        for i, (x, e) in enumerate(pairs):
            if x[0] != "step":
                continue
            run = doc["runs"][x[1]]; s = run["steps"][x[2]]
            prev = pairs[i - 1] if i else None
            o = None
            if prev and prev[0][0] == "step" and prev[0][1] == x[1]:
                o = e["t"] - prev[1]["t"] - s["wall_s"] - COOLDOWN_S
                ovh.append(o)
            t1 = e["t"]; t0 = t1 - s["wall_s"]
            cpu, why = window_cpu(T, Cc, t0, t1)
            cpu_hi, _ = window_cpu(T, Cc, t0 - max(o or 0, 0) - 2, t1)   # widened: whole overhead before + 2 s
            res.append(dict(src=tag, variant=run["variant"], round=run["round"], S=s["sessions"], P=s["pause_median_s"],
                            turns=s["turns"], wall=s["wall_s"], t_save=t1, cpu=cpu, cpu_hi=cpu_hi, why=why, ovh=o,
                            j=s["j_per_turn"], hit=s["cache_hit"], prompt=s["prompt_tokens"], cached=s["cached_tokens"],
                            recomputed=s["recomputed_tokens"], method="mtime"))
        n_steps = sum(len(r["steps"]) for r in doc["runs"])
        unobs = n_steps - sum(1 for x, _ in pairs if x[0] == "step")
        seen = {(x[1], x[2]) for x, _ in pairs if x[0] == "step"}
        unobs_names = [f"{VARIANT_LABEL.get(r['variant'], r['variant'])} r{r['round']} {s_['sessions']}/{s_['pause_median_s']:g} s"
                       for ri, r in enumerate(doc["runs"]) for k, s_ in enumerate(r["steps"]) if (ri, k) not in seen]
        info = dict(events=len(ev), merged=merged, viol=viol, with_final=wf, unobserved=unobs, ovh=ovh, unobs_names=unobs_names,
                    ends=[(x, e) for x, e in pairs if x[0] != "step"])
        return res, info

    main_res, main_info = do_aligned(D["agents.json"], Tm, Cm, Mm, "cpu-time.csv / agents.json")
    off_res, off_info = do_aligned(D["agents-offload.json"], To, Co, Mo, "cpu-time-offload.csv / agents-offload.json")
    # cross-check: the main logger kept running through the offload run (same /proc sum, same clock)
    for x in off_res:
        x["cpu_xcheck"], _ = window_cpu(Tm, Cm, x["t_save"] - x["wall"], x["t_save"])

    # end-of-variant lag (engine stop seen in the CPU log after the run's last step save), per engine, to calibrate
    lag = {"vllm": [], "sglang": []}
    for doc, T, Cc, res in ((D["agents.json"], Tm, Cm, main_res), (D["agents-offload.json"], To, Co, off_res)):
        for r in doc["runs"]:
            last = [x for x in res if x["variant"] == r["variant"] and x["round"] == r["round"]]
            if len(last) == len(r["steps"]) and last:
                ls = max(x["t_save"] for x in last)
                st = [t for t in stops(T, Cc, ls, ls + 300)]
                if st:
                    lag[r["engine"]].append(st[0] - ls)
    ovh_all = [o for o in main_info["ovh"] + off_info["ovh"] if o is not None]
    g_med = statistics.median(ovh_all)

    def do_inferred(doc, T, Cc, tag):
        """No mtime for this JSON in any log: place each run's last step save at (engine stop - calibrated lag), chain
        earlier saves back by wall_s + 20 s + median overhead, and check that every inferred cooldown is quiet."""
        res, notes = [], []
        st = stops(T, Cc, ts(doc["started_at"]), ts(doc["finished_at"]) + 120)
        runs = [r for r in doc["runs"] if r["steps"]]
        if len(st) != len(runs):
            notes.append(f"{tag}: found {len(st)} engine stops for {len(runs)} runs; not aligned")
            return res, notes
        for r, t_stop in zip(runs, st):
            L = statistics.median(lag[r["engine"]])
            t = t_stop - L
            saves = []
            for k in range(len(r["steps"]) - 1, -1, -1):
                saves.append((k, t))
                t = t - r["steps"][k]["wall_s"] - COOLDOWN_S - g_med
            for k, t1 in sorted(saves):
                s = r["steps"][k]
                cpu, why = window_cpu(T, Cc, t1 - s["wall_s"], t1)
                cpu_hi, _ = window_cpu(T, Cc, t1 - s["wall_s"] - 5, t1 + 5)
                q0, _ = window_cpu(T, Cc, t1 + 3, t1 + 17)       # inferred cooldown after this save
                quiet = (q0 / 14) if q0 is not None else None
                res.append(dict(src=tag, variant=r["variant"], round=r["round"], S=s["sessions"], P=s["pause_median_s"],
                                turns=s["turns"], wall=s["wall_s"], t_save=t1, cpu=cpu, cpu_hi=cpu_hi, why=why, ovh=None,
                                j=s["j_per_turn"], hit=s["cache_hit"], prompt=s["prompt_tokens"], cached=s["cached_tokens"],
                                recomputed=s["recomputed_tokens"], method="inferred", quiet=quiet))
        return res, notes

    # validate the inferred method where the truth is known (vLLM 70 GiB, and SGLang in the main run)
    val = []
    inf_off, _ = do_inferred(D["agents-offload.json"], Tm, Cm, "check")
    for a in inf_off:
        b = next(x for x in off_res if (x["variant"], x["round"], x["S"], x["P"]) == (a["variant"], a["round"], a["S"], a["P"]))
        val.append((a["t_save"] - b["t_save"], rel(a["cpu"], b["cpu"])))
    inf_main, _ = do_inferred(D["agents.json"], Tm, Cm, "check")
    for a in inf_main:
        b = [x for x in main_res if (x["variant"], x["round"], x["S"], x["P"]) == (a["variant"], a["round"], a["S"], a["P"])]
        if b and b[0]["cpu"] is not None and a["cpu"] is not None:
            val.append((a["t_save"] - b[0]["t_save"], rel(a["cpu"], b[0]["cpu"]), a["variant"]))
    val = [v if len(v) == 3 else v + ("vllm-offload",) for v in val]
    # reference: engine CPU during true (mtime-aligned) cooldowns, per engine
    qref = {"vllm": [], "sglang": []}
    for x in main_res + off_res:
        q0, _ = window_cpu(Tm, Cm, x["t_save"] + 3, x["t_save"] + 17)
        if q0 is not None:
            qref["sglang" if x["variant"].startswith("sglang") else "vllm"].append(q0 / 14)
    h_res, h_notes = do_inferred(D["agents-host160.json"], Tm, Cm, "cpu-time.csv / agents-host160.json (inferred)")
    n_drops = drops_in(Tm, Cm, Tm[0], Tm[-1])
    R["notes"] += h_notes

    allres = main_res + off_res + h_res
    R["steps"] = allres

    # --- write-up
    out.append("### How the steps were aligned\n")
    out.append(
        "`cpu-time*.csv` log, once per second, the cumulative CPU seconds (user + system) of every process whose command "
        "line contains `vllm` or `sglang` (cpu_log2.py), plus the mtime of one results JSON. agent_energy.py writes that "
        "JSON (a) right after each step (after the sessions join and `wall_s` is measured), then dumps the timeline file "
        "and sleeps a 20 s cooldown before the next step, and (b) once more in `finally` after each variant (engine "
        "stopped, 10 s sleep) — plus a last write when the whole script ends. So:\n\n"
        "1. Collapse mtime changes less than 5 s apart into one save (a large JSON can be caught mid-write: "
        f"{main_info['merged']} such double changes in cpu-time.csv, {off_info['merged']} in cpu-time-offload.csv); the "
        "save time is the mtime itself, not the logger's sample time.\n"
        "2. Build the expected save sequence from the JSON (runs in execution order; per run one save per step, then "
        "one end-of-variant save; runs that failed at start-up have only the end save) and match observed saves to it "
        "from the end backwards, because the main logger started late. Every match is checked: consecutive step saves "
        "must be `wall_s` + 20 s + a small overhead apart, an end save 25-120 s after its run's last step. End-of-variant "
        "saves are matched and skipped.\n"
        "3. Each step's window is [save - wall_s, save]; engine CPU in it is interpolated linearly between samples. An "
        "upper window adds the measured overhead before the step plus 2 s.\n")
    out.append(md_table(["log / JSON", "saves seen", "gap-rule violations", "steps not observed", "overhead between steps s (median, max)",
                         "end saves skipped"],
                        [[r_[0], r_[1]["events"], r_[1]["viol"], r_[1]["unobserved"],
                          f"{statistics.median([o for o in r_[1]['ovh'] if o is not None]):.2f}, {max(o for o in r_[1]['ovh'] if o is not None):.2f}",
                          len(r_[1]["ends"])] for r_ in (("cpu-time.csv / agents.json", main_info), ("cpu-time-offload.csv / agents-offload.json", off_info))]))
    flagged = [x for x in allres if x["cpu"] is None]
    out.append("\n**Steps that cannot be aligned or measured:** " + ("; ".join(
        f"{VARIANT_LABEL.get(x['variant'], x['variant'])} r{x['round']} {x['S']}/{x['P']:g} s ({x['why']})" for x in flagged) or "none")
        + f". Not observed at all: {main_info['unobserved']} step(s) of the main run whose save happened before the logger "
        f"started ({', '.join(main_info['unobs_names'])}). vLLM offload at 80 GB has no steps (start-up failure). "
        "No JSON other than agents.json and agents-offload.json had its mtime logged, so the 160 GB run is aligned by "
        "inference (below), and the gate, LFU/SLRU, cliff and seed runs are not aligned.\n")
    out.append(
        f"\n**Inferred alignment for SGLang 160 GB.** The CPU logger watching agents.json kept running through the "
        f"extra runs. Each engine stop shows as the cumulative CPU dropping by more than half. Calibrated on the "
        f"mtime-aligned runs, the stop is seen {statistics.median(lag['vllm']):.1f} s (vLLM, n={len(lag['vllm'])}) / "
        f"{statistics.median(lag['sglang']):.1f} s (SGLang, n={len(lag['sglang'])}) after the last step's save, and "
        f"consecutive saves are wall_s + 20 s + {g_med:.2f} s apart (median). Each run's saves are placed backwards from "
        f"its engine stop. Validation where the truth is known from mtimes, applying the same inference to the "
        f"vLLM 70 GiB run and to every run of the main matrix: "
        + "; ".join(f"{eng}: save-time error {min(v[0] for v in val if v[2].startswith(eng)):+.1f} to "
                    f"{max(v[0] for v in val if v[2].startswith(eng)):+.1f} s, CPU-seconds error "
                    f"{min(v[1] for v in val if v[2].startswith(eng)):+.2f}% to {max(v[1] for v in val if v[2].startswith(eng)):+.2f}% "
                    f"(n={sum(1 for v in val if v[2].startswith(eng))} steps)" for eng in ("vllm", "sglang"))
        + f". Check on the 160 GB steps: during each inferred 20 s cooldown the engine uses "
        f"{min(x['quiet'] for x in h_res if x['quiet'] is not None):.2f}-{max(x['quiet'] for x in h_res if x['quiet'] is not None):.2f} "
        f"busy cores, the same as during the true cooldowns of SGLang runs ({min(qref['sglang']):.2f}-{max(qref['sglang']):.2f}; SGLang's "
        f"scheduler keeps about one core busy even when idle; vLLM {min(qref['vllm']):.2f}-{max(qref['vllm']):.2f}). For SGLang this "
        f"check is weak (its in-step rate is only ~1.05-1.15 cores); the validation above is the real test. {' '.join(h_notes)}\n")
    out.append(f"\nCPU-seconds are summed from positive increments: the logged sum drops {n_drops} times by more than 0.5 s "
               "without an engine restart (an engine subprocess exiting removes its accumulated CPU from the sum); only a "
               "fall by more than half is treated as a restart.\n")
    xc = [rel(x["cpu_xcheck"], x["cpu"]) for x in off_res if x["cpu"] and x["cpu_xcheck"]]
    out.append(f"\nCross-check: the two loggers ran at the same time during the 70 GiB run; per-step CPU-seconds agree within "
               f"{min(xc):+.2f}% to {max(xc):+.2f}%.\n")

    # per-step table
    rows = []
    for x in sorted(allres, key=lambda x: (x["variant"], x["S"], x["P"], x["round"])):
        cps = x["cpu"] / x["turns"] if x["cpu"] is not None else None
        rows.append([VARIANT_LABEL.get(x["variant"], x["variant"]), x["round"], x["S"], f"{x['P']:g} s", x["method"], utc(x["t_save"]),
                     x["turns"], f(x["wall"], 1), f(x["cpu"], 1), f(x["cpu"] / x["wall"] if x["cpu"] is not None else None, 2),
                     f(cps, 3), f(x["cpu_hi"] / x["turns"] if x["cpu_hi"] is not None else None, 3), x["why"] or ""])
    out.append("\n### Engine CPU per step\n")
    out.append(md_table(["variant", "round", "sessions", "pause", "alignment", "save (UTC)", "turns", "wall s", "engine CPU s",
                         "avg busy cores", "CPU s / turn", "CPU s / turn (upper window)", "flag"], rows))

    # comparison offload vs GPU-only, per cell (mean over rounds of what is available)
    def cell(variant, S, P, key="cpu"):
        xs = [x for x in allres if x["variant"] == variant and x["S"] == S and x["P"] == P and x[key] is not None]
        if not xs:
            return None, 0
        return sum(x[key] for x in xs) / sum(x["turns"] for x in xs), len(xs)

    comp, rows = [], []
    pairs = [("vllm-offload", "vllm-gpu", C["off"], "vllm-offload70", "vLLM 70 GiB vs vLLM GPU-only"),
             ("sglang-offload", "sglang-gpu", C["main"], "sglang-offload", "SGLang 80 GB vs SGLang GPU-only"),
             ("sglang-offload160", "sglang-gpu", C["h"], "sglang-offload160", "SGLang 160 GB (inferred) vs SGLang GPU-only")]
    for ov, gv, cc, ckey, lab in pairs:
        for S in (16, 48, 96):
            for P in (5.0, 30.0):
                co, no = cell(ov, S, P); cg, ng = cell(gv, S, P)
                co_hi, _ = cell(ov, S, P, "cpu_hi")
                Jo = cell_mean(cc, (ckey, S, P), "j_per_turn"); Jg = cell_mean(C["main"], (gv, S, P), "j_per_turn")
                saved = Jg - Jo if (Jg is not None and Jo is not None) else None
                xs = [x for x in allres if x["variant"] == ov and x["S"] == S and x["P"] == P]
                # host-memory/PCIe traffic bound: every recomputed token written to host + every cached token read back
                byt = mean([(x["recomputed"] + x["cached"]) * KV_BYTES_PER_TOKEN / x["turns"] for x in xs]) if xs else None
                extra = (co - cg) if (co is not None and cg is not None) else None
                cpu_j_tot = W_PER_CORE * co_hi if co_hi is not None else None
                cpu_j_extra = W_PER_CORE * extra if extra is not None else None
                mem_j = J_PER_BYTE * byt if byt is not None else None
                ratio_tot = 100 * cpu_j_tot / saved if (cpu_j_tot is not None and saved and saved > 0) else None
                ratio_ex = 100 * (max(cpu_j_extra, 0) + (mem_j or 0)) / saved if (cpu_j_extra is not None and saved and saved > 0) else None
                comp.append(dict(lab=lab, ov=ov, S=S, P=P, co=co, cg=cg, no=no, ng=ng, saved=saved, cpu_j_tot=cpu_j_tot,
                                 cpu_j_extra=cpu_j_extra, mem_j=mem_j, ratio_tot=ratio_tot, ratio_ex=ratio_ex))
                rows.append([lab, S, f"{P:g} s", f"{f(co,3)} (n={no})", f"{f(cg,3)} (n={ng})", f(extra, 3),
                             f(cpu_j_tot, 1), f(cpu_j_extra, 1), f(mem_j, 1), f(saved, 1),
                             (f(ratio_ex, 1) + "%") if ratio_ex is not None else ("GPU saving <= 0" if saved is not None and cpu_j_extra is not None else "n/a"),
                             (f(ratio_tot, 1) + "%") if ratio_tot is not None else "n/a"])
    R["comp"] = comp
    out.append("\n### Offload vs GPU-only: engine CPU per turn and the energy bound\n")
    out.append(
        f"CPU energy bound = {W_PER_CORE:g} W per busy core x engine CPU-seconds (an upper bound on the engine's CPU energy "
        "if a busy core draws at most 10 W; it ignores package/uncore power that would be drawn anyway). 'Extra' = offload "
        "minus GPU-only CPU-seconds per turn in the same cell (rounds pooled). The host-memory term is an **assumption**, "
        f"not a measurement: every recomputed token written to host and every cached token read back, at "
        f"{KV_BYTES_PER_TOKEN:,} bytes/token and {J_PER_BYTE*1e9:g} nJ/byte. GPU saved = GPU-only J/turn - offload J/turn. "
        "H5 asks for (extra CPU + host memory) < 10% of the GPU saving; the last column is the looser test that charges "
        "the offload engine's entire CPU time.\n")
    out.append(md_table(["comparison", "sessions", "pause", "offload CPU s/turn", "GPU-only CPU s/turn", "extra CPU s/turn",
                         "bound: all offload CPU J/turn (upper window)", "bound: extra CPU J/turn", "host-memory J/turn (assumed)",
                         "GPU J/turn saved", "(extra CPU + memory) / GPU saved", "all offload CPU / GPU saved"], rows))
    return "\n".join(out), R, dict(main=main_info, off=off_info, lag=lag, g=g_med, val=val)


# ----------------------------------------------------------------------------------------------------------------------
# Capacities from server logs (data, read as text)
# ----------------------------------------------------------------------------------------------------------------------
def capacities():
    caps = {}
    pats = [("vllm_gpu", "main/vllm-gpu-agent-r1-server.log", r"GPU KV cache size: ([\d,]+) tokens"),
            ("sglang_gpu", "main/sglang-gpu-agent-r1-server.log", r"max_total_num_tokens=(\d+)"),
            ("sglang_host80", "main/sglang-offload-agent-r1-server.log", r"KV host pool: (\d+) tokens"),
            ("sglang_host160", "extra/sglang-offload160-agent-r1-server.log", r"KV host pool: (\d+) tokens")]
    for k, fn, p in pats:
        try:
            with open(os.path.join(DATA, fn), errors="ignore") as fh:
                mm = re.search(p, fh.read())
            caps[k] = int(mm.group(1).replace(",", "")) if mm else None
        except OSError:
            caps[k] = None
    return caps


# ----------------------------------------------------------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------------------------------------------------------
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                         "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7})
    return plt


def plain_log(ax, axis, ticks):
    """Log axis with a few plain-number ticks instead of 10^x labels."""
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    return p


def fig_cliff(plt, C, M):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    series = [("vLLM", "vllm-gpu", [16, 24, 32, 40, 48, 96]), ("SGLang", "sglang-gpu", [16, 48, 96])]
    i = 0
    for eng, v, Ss in series:
        for P, ls in ((5.0, "-"), (30.0, "--")):
            xs, hit, j = [], [], []
            for S in Ss:
                cc = C["cl"] if S in (24, 32, 40) else C["main"]
                xs.append(S); hit.append(100 * cell_mean(cc, (v, S, P), "cache_hit")); j.append(cell_mean(cc, (v, S, P), "j_per_turn"))
            col = PAL[0] if eng == "vLLM" else PAL[1]
            mk = "o" if P == 5.0 else "s"
            ax[0].plot(xs, hit, ls, color=col, marker=mk, label=f"{eng} GPU-only, pause {P:g} s")
            ax[1].plot(xs, j, ls, color=col, marker=mk, label=f"{eng} GPU-only, pause {P:g} s")
            i += 1
    for P, col in ((5.0, PAL[0]), (30.0, PAL[2])):
        J = [cell_mean(C["main"], ("vllm-nocache", S, P), "j_per_turn") for S in (16, 48, 96)]
        ax[1].plot([16, 48, 96], J, ":", color=INK2, marker="x", label=f"vLLM no cache, pause {P:g} s" if P == 5.0 else "vLLM no cache, pause 30 s")
    ax[0].set(xlabel="concurrent agent sessions", ylabel="cache hit (% of prompt tokens)", title="Cache hit collapses with sessions")
    ax[1].set(xlabel="concurrent agent sessions", ylabel="GPU energy per turn (J)", title="Energy per turn rises to the no-cache level")
    for a in ax[:2]:
        a.set_xticks([16, 24, 32, 40, 48, 96])
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    pts = M["cliff"]["points"]
    for P, col, mk in ((5.0, PAL[0], "o"), (30.0, PAL[2], "s")):
        xs = [p["wc"] for p in pts if p["P"] == P and p["src"].startswith("vLLM")]
        ys = [100 * p["rhit_turn"] for p in pts if p["P"] == P and p["src"].startswith("vLLM")]
        ax[2].scatter(xs, ys, color=col, marker=mk, s=40, label=f"vLLM, pause {P:g} s", zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
        xs = [p["wc"] for p in pts if p["P"] == P and p["src"].startswith("SGLang")]
        ys = [100 * p["rhit_turn"] for p in pts if p["P"] == P and p["src"].startswith("SGLang")]
        ax[2].scatter(xs, ys, facecolor="none", edgecolor=col, marker=mk, s=60, label=f"SGLang, pause {P:g} s", zorder=3)
    for x, lab in ((1.0, "A: fixed cycle (1.0)"), (1.58, "B: random order (1.58)"), (1.9, "Metastability (1.9)")):
        ax[2].axvline(x, color=INK2, lw=1, ls=":")
        ax[2].text(x, 101, lab, rotation=90, va="top", ha="right", fontsize=8, color=INK2)
    ax[2].set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax[2].xaxis.set_major_locator(FixedLocator([0.75, 1, 1.25, 1.5, 2, 3, 4]))
    ax[2].xaxis.set_minor_locator(NullLocator())
    ax[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[2].set(xlabel=f"working set / capacity W/C (C = {C_GPU:,} vLLM, {251423:,} SGLang; log scale)", ylabel="resumed turns that hit (%)",
              title="Resumed-turn hit vs W/C (color = pause)")
    ax[2].legend(fontsize=8, loc="center right")
    fig.suptitle("Cliff: GPU-only prefix cache, 150 s steps, Qwen3-32B on H200 NVL (mean of 2 rounds)", fontsize=11)
    return save(fig, "cliff.png")


def fig_ram(plt, C, caps):
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    bars = [("SGLang GPU-only", C["main"], "sglang-gpu"), ("SGLang 80 GB host", C["main"], "sglang-offload"),
            ("SGLang 160 GB host", C["h"], "sglang-offload160"), ("vLLM GPU-only", C["main"], "vllm-gpu"),
            ("vLLM 70 GiB host", C["off"], "vllm-offload70")]
    w = 0.16
    for ci, P in enumerate((5.0, 30.0)):
        for bi, (lab, cc, v) in enumerate(bars):
            for ri, (field, ylab) in enumerate((("j_per_turn", "GPU energy per turn (J)"), ("turns", "turns completed in 150 s"))):
                xs, ys, lo, hi = [], [], [], []
                for gi, S in enumerate((48, 96)):
                    vals = [s[field] for s in cc[(v, S, P)].values()]
                    xs.append(gi + (bi - 2) * w); ys.append(mean(vals)); lo.append(mean(vals) - min(vals)); hi.append(max(vals) - mean(vals))
                a = ax[ri][ci]
                a.bar(xs, ys, w * 0.9, color=PAL[bi], label=lab, yerr=[lo, hi], capsize=2, ecolor=INK2)
                for x_, y_ in zip(xs, ys):
                    a.text(x_, y_, f"{y_:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
                a.set_ylabel(ylab)
                a.set_title(f"pause {P:g} s")
    for a in ax[1]:
        a.set_xticks([0, 1]); a.set_xticklabels(["48 sessions", "96 sessions"])
    h_, l_ = ax[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.1)
    fig.suptitle(f"Host RAM sizing: GPU-only vs host tiers (host pool tokens: 80 GB = {caps.get('sglang_host80') or 305176:,}, "
                 f"160 GB = {caps.get('sglang_host160') or 610352:,}; bars = mean of 2 rounds, whiskers = round range)", fontsize=10)
    return save(fig, "ram_sizing.png")


def fig_transfer(plt, D, M):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    floor = 1e-6
    groups = [("recompute (both variants)", lambda v, ph: ph == "recompute", PAL[1], "o"),
              ("GPU-only control after flush (recompute)", lambda v, ph: v == "sglang-gpu-control" and ph == "after_flush", PAL[3], "D"),
              ("restore from 160 GB host tier", lambda v, ph: v == "sglang-host160" and ph == "after_flush", PAL[0], "s"),
              ("GPU prefix-cache hit (both variants)", lambda v, ph: ph == "gpu_hit", PAL[2], "^")]
    for lab, sel, col, mk in groups:
        pts = []
        for r in D["transfer.json"]["runs"]:
            for s in r["steps"]:
                for ph in ("recompute", "gpu_hit", "after_flush"):
                    if sel(r["variant"], ph):
                        x = s[ph]
                        pts.append((x["prompt_tokens"] / x["requests"], x["above_idle_j_per_prompt_token"], s["length"]))
        ax.scatter([p[0] for p in pts], [max(p[1], floor) for p in pts], color=col, marker=mk, s=30, alpha=0.75, label=lab,
                   edgecolor="#fcfcfb", linewidth=0.8)
        byL = {}
        for p in pts:
            byL.setdefault(p[2], []).append(p)
        xm = [mean([p[0] for p in v]) for _, v in sorted(byL.items())]
        ym = [max(mean([p[1] for p in v]), floor) for _, v in sorted(byL.items())]
        ax.plot(xm, ym, color=col, lw=1.5)
    k = M["flop"]["k_act"]
    Ls = [4000 + i * 250 for i in range(70)]
    ax.plot(Ls, [k * flops_per_token(L) for L in Ls], color=INK2, ls="--", lw=1.2, label="FLOP model, J/FLOP fitted at the 8k point")
    ax.axhline(floor, color=GRID, lw=1)
    ax.text(4200, floor * 1.15, "values <= 0 drawn at 1e-6", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_xscale("log")
    plain_log(ax, "x", [5000, 10000, 20000])
    ax.set(xlabel="actual prompt tokens per request (log scale)", ylabel="above-idle GPU energy per prompt token (J, log scale)",
           title="Transfer vs recompute: 8 prompts at once, 3 repetitions (SGLang, H200 NVL)")
    ax.legend(fontsize=8, loc="center right")
    return save(fig, "transfer_vs_recompute.png")


def fig_gate(plt, C, fair, ref):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    cells = [(48, 5.0), (48, 30.0), (96, 5.0), (96, 30.0)]
    labs = [f"{S} sess.\n{P:g} s" for S, P in cells]
    w = 0.26
    series = [("vLLM GPU-only", None, PAL[0]), ("gate 16x4", "vllm-admit16x4", PAL[1]), ("gate 16x8", "vllm-admit16x8", PAL[2])]
    for bi, (lab, v, col) in enumerate(series):
        J, served, lat, latc = [], [], [], []
        for S, P in cells:
            if v is None:
                rr = ref[(S, P)]
                J.append(mean([x["j"] for x in rr])); served.append(100.0); lat.append(mean([x["lat90"] for x in rr])); latc.append(None)
            else:
                rr = fair[(v, S, P)]
                J.append(mean([x["j"] for x in rr])); served.append(100 * mean([x["served"] / x["S"] for x in rr]))
                lat.append(mean([x["lat90"] for x in rr])); latc.append(mean([x["lat90c"] for x in rr]))
        xs = [i + (bi - 1) * w for i in range(len(cells))]
        ax[0].bar(xs, J, w * 0.9, color=col, label=lab)
        ax[1].bar(xs, served, w * 0.9, color=col, label=lab)
        ax[2].bar(xs, lat, w * 0.9, color=col, label=lab + " (served turns)")
        if v is not None:
            ax[2].scatter(xs, latc, marker="_", s=250, color=INK, zorder=4, label="incl. never-served (censored, lower bound)" if bi == 1 else None)
        for a, ys in ((ax[0], J), (ax[1], served), (ax[2], lat)):
            for x_, y_ in zip(xs, ys):
                a.text(x_, y_, f"{y_:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax[0].set(ylabel="GPU energy per turn (J)", title="Energy per turn")
    ax[1].set(ylabel="sessions served at least once (%)", title="Fairness: sessions served in a 150 s step")
    ax[2].set(ylabel="p90 wait + TTFT, all turns (s)", title="Latency to first token incl. the wait for a slot")
    for a in ax:
        a.set_xticks(range(len(cells))); a.set_xticklabels(labs)
    h_, l_ = ax[2].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle("Admission gate (16 slots, yield after q turns) vs GPU-only, vLLM, mean of 2 rounds", fontsize=11)
    return save(fig, "gate_fairness.png")


def fig_p2(plt, C):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    vs = [("vllm-default", PAL[0], "-", "o"), ("vllm-tuned", PAL[0], "--", "s"), ("sglang-default", PAL[1], "-", "o"),
          ("sglang-tuned", PAL[1], "--", "s")]
    for wi, wl in enumerate(("chat", "rag")):
        a = ax[wi]
        for v, col, ls, mk in vs:
            ks = sorted(k for k in C["p2"] if k[0] == v and k[1] == wl)
            x = [mean([s["ttft_p90"] for s in C["p2"][k].values()]) for k in ks]
            y = [mean([s["j_per_output_token"] for s in C["p2"][k].values()]) for k in ks]
            a.plot(x, y, ls, color=col, marker=mk, label=v)
            if v == "sglang-tuned" or v == "vllm-default":
                for k, x_, y_ in zip(ks, x, y):
                    a.annotate(f"{k[2]:g}/s", (x_, y_), textcoords="offset points", xytext=(5, 4 if v == "sglang-tuned" else -10),
                               fontsize=7, color=INK2)
        a.set_xscale("log"); a.set_yscale("log")
        plain_log(a, "x", [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
        plain_log(a, "y", [0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 7])
        a.set(xlabel="TTFT p90 (s, log scale)", ylabel="GPU energy per output token (J, log scale)",
              title=f"{wl}: open-loop Poisson arrivals, labels = request rate")
        a.legend(fontsize=8)
    fig.suptitle("P2 energy-latency knee: energy per token falls with load until queueing explodes TTFT (mean of 2 rounds)", fontsize=11)
    return save(fig, "p2_knee.png")


# ----------------------------------------------------------------------------------------------------------------------
# 6. Summary
# ----------------------------------------------------------------------------------------------------------------------
def summary(D, C, M, fair, ref, CPU, caps, cpuinfo):
    m, g, h, off, ev, cl = C["main"], C["gate"], C["h"], C["off"], C["ev"], C["cl"]
    J = lambda cc, k: cell_mean(cc, k, "j_per_turn")
    H = lambda cc, k: cell_mean(cc, k, "cache_hit")
    R90 = lambda cc, k: cell_mean(cc, k, "resume_ttft_p90")
    L = []
    w = L.append
    w("# Summary: KV-cache energy study, analysis of the recorded results\n")
    w("Generated by `analyze.py` from all-results.zip and extra-results.zip. GPU board energy only (NVML). Details: "
      "tables.md (every cell, both rounds), analysis.md (fairness, models, CPU bound), figures/. Hypotheses H11-H16 and "
      "H17/H18 have no data in these two zips and are not judged here.\n")
    w("## Verdicts\n")

    # H1
    p2 = C["p2"]
    diffs_def, diffs_tun = [], []
    for wl in ("chat", "rag"):
        for rate in sorted({k[2] for k in p2 if k[1] == wl}):
            a = mean([s["j_per_output_token"] for s in p2[("vllm-default", wl, rate)].values()])
            b = mean([s["j_per_output_token"] for s in p2[("sglang-default", wl, rate)].values()])
            c = mean([s["j_per_output_token"] for s in p2[("vllm-tuned", wl, rate)].values()])
            d = mean([s["j_per_output_token"] for s in p2[("sglang-tuned", wl, rate)].values()])
            diffs_def.append((rel(b, a), wl, rate)); diffs_tun.append((rel(d, c), wl, rate))
    wd = max(diffs_def, key=lambda x: abs(x[0])); wt = max(diffs_tun, key=lambda x: abs(x[0]))
    jmax = max(mean([s["j_per_output_token"] for s in v.values()]) for v in p2.values())
    jmin = min(mean([s["j_per_output_token"] for s in v.values()]) for v in p2.values())
    rag8 = lambda v, f_: mean([s[f_] for s in p2[(v, "rag", 8.0)].values()])
    vt = [abs(rel(mean([s["j_per_output_token"] for s in p2[("vllm-tuned",) + k[1:]].values()]),
                  mean([s["j_per_output_token"] for s in p2[k].values()]))) for k in p2 if k[0] == "vllm-default"]
    w(f"- **H1 (engines within 3%): partly supported.** Default vs default, SGLang/vLLM J per output token differ by at "
      f"most {wd[0]:+.1f}% ({wd[1]} {wd[2]:g}/s); tuned vs tuned by up to {wt[0]:+.1f}% ({wt[1]} {wt[2]:g}/s) — outside "
      f"3%, because SGLang's tuned flags cut RAG 8/s to {rag8('sglang-tuned','j_per_output_token'):.3f} J/token (TTFT p90 "
      f"{rag8('sglang-tuned','ttft_p90'):.1f} s vs {rag8('sglang-default','ttft_p90'):.1f} s default) while vLLM tuning "
      f"changes little (mean |difference| {mean(vt):.2f}% over the 12 cells, largest {max(vt):.1f}% at RAG 8/s, "
      f"{rel(rag8('vllm-tuned','j_per_output_token'), rag8('vllm-default','j_per_output_token')):+.1f}%; the plan says 0.1%). "
      f"Workload range {jmax/jmin:.1f}x (> 5x: supported).\n")

    # H2
    e2 = {}
    for fn in ("e2-len1600.json", "e2-len12600.json"):
        for e in D[fn]["engines"]:
            by = {(r["workload"], r["users"]): r for r in e["rows"]}
            for u in (1, 64):
                e2[(fn, e["engine"], u)] = by[("long_miss", u)]["joules_per_token"] / by[("long_hit", u)]["joules_per_token"]
    w(f"- **H2 (miss penalty grows with load and length): supported (one round only).** Miss/hit J per output token at 1 user: "
      f"{e2[('e2-len1600.json','vllm',1)]:.2f}x/{e2[('e2-len1600.json','sglang',1)]:.2f}x (2k, vLLM/SGLang), "
      f"{e2[('e2-len12600.json','vllm',1)]:.2f}x/{e2[('e2-len12600.json','sglang',1)]:.2f}x (16k) — all <= 1.5x; at 64 users "
      f"{e2[('e2-len1600.json','vllm',64)]:.1f}x/{e2[('e2-len1600.json','sglang',64)]:.1f}x (2k) and "
      f"{e2[('e2-len12600.json','vllm',64)]:.1f}x/{e2[('e2-len12600.json','sglang',64)]:.1f}x (16k, >= 5x). The 8k point is "
      f"part 1 and not in these zips. E2 has no second round.\n")

    # H3
    s80 = ("sglang-offload", 48, 5.0); sg = ("sglang-gpu", 48, 5.0)
    v70 = ("vllm-offload70", 48, 5.0); vg = ("vllm-gpu", 48, 5.0)
    w(f"- **H3 (offload >= 20% less energy and >= 2x lower resume TTFT p90 at 48 sessions): not supported.** SGLang 80 GB: "
      f"{J(m,s80):.0f} vs {J(m,sg):.0f} J/turn ({rel(J(m,s80),J(m,sg)):+.1f}%), hit {100*H(m,s80):.1f}%, resume TTFT p90 "
      f"{R90(m,s80):.1f} vs {R90(m,sg):.1f} s ({R90(m,sg)/R90(m,s80):.2f}x). vLLM 70 GiB: {J(off,v70):.0f} vs {J(m,vg):.0f} "
      f"({rel(J(off,v70),J(m,vg)):+.1f}%), hit {100*H(off,v70):.1f}%. vLLM at 80 GB never ran (/dev/shm). 'No pressure "
      f"(16 sessions) within noise' is **not supported** either: offload is "
      f"{rel(J(m,('sglang-offload',16,5.0)),J(m,('sglang-gpu',16,5.0))):+.1f}% (SGLang) and "
      f"{rel(J(off,('vllm-offload70',16,5.0)),J(m,('vllm-gpu',16,5.0))):+.1f}% (vLLM) per turn at 16 sessions/5 s with "
      f"~89% vs ~85% hit, while rounds repeat within ~1%. 'Benefit shrinks at 96': yes "
      f"({rel(J(m,('sglang-offload',96,5.0)),J(m,('sglang-gpu',96,5.0))):+.1f}% SGLang 80 GB).\n")

    # H4
    def sav(cc, ov, gv, S, P):
        return rel(J(cc, (ov, S, P)), J(m, (gv, S, P)))
    w(f"- **H4 (offload benefit larger at 30 s pauses): not supported overall (mixed).** Change in J/turn vs GPU-only, "
      f"48 sessions, 5 s -> 30 s: SGLang 80 GB {sav(m,'sglang-offload','sglang-gpu',48,5.0):+.1f}% -> "
      f"{sav(m,'sglang-offload','sglang-gpu',48,30.0):+.1f}% (same); SGLang 160 GB {sav(h,'sglang-offload160','sglang-gpu',48,5.0):+.1f}% -> "
      f"{sav(h,'sglang-offload160','sglang-gpu',48,30.0):+.1f}% (smaller); vLLM 70 GiB {sav(off,'vllm-offload70','vllm-gpu',48,5.0):+.1f}% -> "
      f"{sav(off,'vllm-offload70','vllm-gpu',48,30.0):+.1f}% (larger, from a 1.7% -> 7.9% hit).\n")

    # H5
    comp = {(c["ov"], c["S"], c["P"]): c for c in CPU["comp"]}
    c160 = comp[("sglang-offload160", 48, 5.0)]; c80 = comp[("sglang-offload", 48, 5.0)]; c70 = comp[("vllm-offload", 48, 5.0)]
    w(f"- **H5 (offload's own CPU + host-memory energy < 10% of the GPU energy saved): supported where offload saves real "
      f"energy, as a bound only; not supported where it saves little.** SGLang 160 GB, 48/5 s (inferred alignment): engine CPU "
      f"{f(c160['co'],2)} vs {f(c160['cg'],2)} s/turn for GPU-only, i.e. *less* per turn (SGLang's scheduler keeps ~1 core busy "
      f"whatever it does, so per-turn CPU follows wall time per turn, and offload completes more turns); the bound is then set "
      f"by the assumed host-memory traffic, {f(c160['mem_j'],1)} J/turn at 1 nJ/byte, against {f(c160['saved'],0)} J of GPU "
      f"energy saved ({f(c160['ratio_ex'],1)}%; charging the offload engine's entire CPU time at 10 W/core: {f(c160['ratio_tot'],1)}%). "
      f"SGLang 80 GB, 48/5 s: {f(c80['ratio_ex'],1)}% of a {f(c80['saved'],0)} J saving. vLLM 70 GiB, 48/5 s: GPU saving only "
      f"{f(c70['saved'],1)} J/turn, ratio {f(c70['ratio_ex'],0)}%; at 96/5 s {f(comp[('vllm-offload',96,5.0)]['ratio_ex'],0)}%. "
      f"At 16 sessions the ratio is 12-14% only because of the assumed memory term. Host CPU and DRAM were not metered (no "
      f"RAPL); a lower J/byte would move every ratio down.\n")

    # H6
    rows6 = []
    for S in (48, 96):
        for P in (5.0, 30.0):
            for v in ("vllm-admit16x4", "vllm-admit16x8"):
                rows6.append((S, P, v, J(g, (v, S, P)), J(m, ("vllm-gpu", S, P)), H(g, (v, S, P))))
    fr = {k: v for k, v in fair.items()}
    nev = [x["never"] for v in fr.values() for x in v if x["S"] == 48]
    nev96 = [x["never"] for v in fr.values() for x in v if x["S"] == 96]
    w90 = [x["fw90"] for k, v in fr.items() if k[0] == "vllm-admit16x4" and k[1] == 48 for x in v]
    a90 = {k: mean([x["w90"] for x in v]) for k, v in fr.items()}
    l90 = {k: mean([x["lat90"] for x in v]) for k, v in fr.items()}
    l90c = {k: mean([x["lat90c"] for x in v]) for k, v in fr.items()}
    gref = {k: mean([x["lat90"] for x in v]) for k, v in ref.items()}
    w(f"- **H6 (gate: >= 30% hit and >= 25% less energy at 48 sessions, both pauses): supported at 5 s, not at 30 s.** "
      + "; ".join(f"{VARIANT_LABEL[v][5:]} {S}/{P:g}s {Jg:.0f} vs {Jb:.0f} J ({rel(Jg,Jb):+.0f}%), hit {100*hh:.0f}%"
                  for S, P, v, Jg, Jb, hh in rows6 if S == 48)
      + f". At 96/5 s: {J(g,('vllm-admit16x4',96,5.0)):.0f}/{J(g,('vllm-admit16x8',96,5.0)):.0f} vs {J(m,('vllm-gpu',96,5.0)):.0f} J. "
      f"**Latency cost:** sessions never served in a 150 s step: {min(nev)}-{max(nev)} of 48 and {min(nev96)}-{max(nev96)} of 96; "
      f"no session was ever re-admitted. Wait p90 over all turns, q=4: {a90[('vllm-admit16x4',48,5.0)]:.0f} s (48/5 s), "
      f"{a90[('vllm-admit16x4',96,5.0)]:.0f} s (96/5 s); first-turn wait p90 {min(w90):.0f}-{max(w90):.0f} s at 48 sessions; the "
      f"recorded resume-turn gate wait is 0 s in every step. p90 latency to first token (wait + TTFT, served turns) at 48/5 s: "
      f"{l90[('vllm-admit16x4',48,5.0)]:.0f} s (q=4) and {l90[('vllm-admit16x8',48,5.0)]:.0f} s (q=8) vs {gref[(48,5.0)]:.0f} s "
      f"GPU-only; with never-served sessions counted (censored) {l90c[('vllm-admit16x4',48,5.0)]:.0f} / "
      f"{l90c[('vllm-admit16x8',48,5.0)]:.0f} s. Part of the per-turn saving comes from serving a warm subset.\n")

    # H7
    w(f"- **H7 (LFU >= 20% hit at 48; SLRU < 5%; 16-session energy within 5%): LFU not supported, SLRU supported.** "
      f"48/5 s hit: LFU {100*H(ev,('sglang-lfu',48,5.0)):.2f}%, SLRU {100*H(ev,('sglang-slru',48,5.0)):.2f}%, LRU "
      f"{100*H(m,('sglang-gpu',48,5.0)):.2f}%; J/turn {J(ev,('sglang-lfu',48,5.0)):.1f}/{J(ev,('sglang-slru',48,5.0)):.1f} vs "
      f"{J(m,('sglang-gpu',48,5.0)):.1f}. 16/5 s: LFU {rel(J(ev,('sglang-lfu',16,5.0)),J(m,('sglang-gpu',16,5.0))):+.1f}%, "
      f"SLRU {rel(J(ev,('sglang-slru',16,5.0)),J(m,('sglang-gpu',16,5.0))):+.1f}%; 16/30 s "
      f"{rel(J(ev,('sglang-lfu',16,30.0)),J(m,('sglang-gpu',16,30.0))):+.1f}% / "
      f"{rel(J(ev,('sglang-slru',16,30.0)),J(m,('sglang-gpu',16,30.0))):+.1f}% (within 5%).\n")

    # H8
    cr = M["cliff"]["cross"]
    w(f"- **H8 (cliff model A at N*~23 / W/C 1, or B at N*~38 / W/C 1.58): neither fits both pauses.** Step hit, 5 s: "
      + ", ".join(f"{S}: {100*H(cl if S in (24,32,40) else m, ('vllm-gpu',S,5.0)):.2f}%" for S in (16, 24, 32, 40, 48))
      + "; 30 s: " + ", ".join(f"{S}: {100*H(cl if S in (24,32,40) else m, ('vllm-gpu',S,30.0)):.2f}%" for S in (16, 24, 32, 40, 48))
      + f". Resumed-turn hit falls to half its 16-session level at W/C ~{f(cr[5.0][0],2)} (5 s) and ~{f(cr[30.0][0],2)} (30 s); "
      f"the last W/C with hits > 2% is {f(cr[5.0][1],2)} (5 s) and {f(cr[30.0][1],2)} (30 s). Both step-function models "
      "miss the partial-hit points by tens of percentage points (models.md (d)). Model A is close at 5 s pauses, and the "
      "30 s collapse lies beyond model B's 1.58.\n")

    # H9
    w(f"- **H9 (160 GB: >= 50% hit and >= 20% less energy at 48, both pauses; < 10% benefit at 96): 48 supported, 96 not.** "
      f"48/5 s: hit {100*H(h,('sglang-offload160',48,5.0)):.1f}%, {J(h,('sglang-offload160',48,5.0)):.1f} J "
      f"({sav(h,'sglang-offload160','sglang-gpu',48,5.0):+.0f}%), {cell_mean(h,('sglang-offload160',48,5.0),'turns'):.0f} turns vs "
      f"{cell_mean(m,('sglang-gpu',48,5.0),'turns'):.0f}; 48/30 s: hit {100*H(h,('sglang-offload160',48,30.0)):.1f}%, "
      f"{sav(h,'sglang-offload160','sglang-gpu',48,30.0):+.0f}%. 96 sessions: {sav(h,'sglang-offload160','sglang-gpu',96,5.0):+.0f}% "
      f"(5 s) and {sav(h,'sglang-offload160','sglang-gpu',96,30.0):+.0f}% (30 s), larger than the < 10% predicted.\n")

    # H10
    tr = C["tr"]
    def tj(v, L_, ph):
        return mean([x["above_idle_j_per_prompt_token"] for x in tr[(v, L_, ph)].values()])
    rat = {L_: mean([tj("sglang-host160", L_, "recompute"), tj("sglang-gpu-control", L_, "recompute")]) / tj("sglang-host160", L_, "after_flush")
           for L_ in (4000, 8000, 16000)}
    ctl = [rel(tr[("sglang-gpu-control", L_, "after_flush")][r]["energy_j"], tr[("sglang-gpu-control", L_, "recompute")][r]["energy_j"])
           for L_ in (4000, 8000, 16000) for r in ("rep1", "rep2", "rep3")]
    w(f"- **H10 (restore <= 1/10 of recompute; GPU hit <= 1/20; control = recompute within 10%): supported.** Recompute/restore "
      f"= {rat[4000]:.0f}x / {rat[8000]:.0f}x / {rat[16000]:.0f}x at nominal 4k/8k/16k (actual ~5.0k/10.0k/20.3k tokens). GPU "
      f"hit <= {max(max(x['above_idle_j_per_prompt_token'] for x in tr[(v_,L_,'gpu_hit')].values()) for v_ in ('sglang-host160','sglang-gpu-control') for L_ in (4000,8000,16000)):.5f} J/token. "
      f"Control after flush vs recompute energy: {min(ctl):+.1f}% to {max(ctl):+.1f}%.\n")

    sd = C["sd"]
    w(f"- **Seeds replicate:** s1/s2 at 16/5 s hit {100*sd[('vllm-gpu',16,5.0)]['s1']['cache_hit']:.1f}%/"
      f"{100*sd[('vllm-gpu',16,5.0)]['s2']['cache_hit']:.1f}%, 48/5 s cache vs no cache "
      f"{rel(J(sd,('vllm-gpu',48,5.0)),J(sd,('vllm-nocache',48,5.0))):+.1f}% J/turn.\n")

    tn = []
    for eng in ("vllm", "sglang"):
        for S in (48, 96):
            for P in (5.0, 30.0):
                tn.append(rel(J(m, (f"{eng}-gpu", S, P)), J(m, (f"{eng}-nocache", S, P))))
    w(f"- **Thrashing = no cache (theory sec. 3): confirmed.** GPU-only vs no cache at 48/96 sessions: vLLM "
      f"{min(tn[:4]):+.2f}% to {max(tn[:4]):+.2f}%, SGLang {min(tn[4:]):+.2f}% to {max(tn[4:]):+.2f}% J/turn.\n")

    # Models
    w("## Models\n")
    fa = M["offload"]
    bad = [x for x in fa if abs(x["err"]) > 2]
    w(f"- Offload model, f = {M['f']:.3f} (fitted on 160 GB 48/5 s): " + "; ".join(
        f"{x['lab']} {x['S']}/{x['P']:g}s {x['pred']:.0f} vs {x['meas']:.0f} ({sgn(x['err'])})" for x in fa) + ". "
      + ("**Fails** (> 2%) at: " + ", ".join(f"{x['lab']} {x['S']}/{x['P']:g} s ({sgn(x['err'])})" for x in bad) + ". " if bad else "")
      + "Cells with small dh test little in total J (the prediction stays near E_thrash), but even the predicted *saving* "
      "is within 4% wherever the measured saving exceeds 5 J, except 160 GB at 48/30 s, where the saving is over-predicted by "
      + f"{[sgn(rel(x['sav_p'],x['sav_m']),0) for x in fa if x['lab']=='SGLang 160 GB' and x['S']==48][0]} (at 30 s pauses a "
      "larger share of turn energy is idle/decode, so a single f does not carry over).\n")
    gb = M["gate"]
    w("- Gate model E(q): with the plan's constants (1,030/510 J) errors are "
      + ", ".join(f"{x['S']}/{x['P']:g}s q={x['q']} {sgn(rel(x['p_plan'],x['meas']),0)}" for x in gb)
      + "; with measured same-pause E_cold/E_warm: " + ", ".join(f"{sgn(rel(x['p_meas'],x['meas']),0)}" for x in gb)
      + ". The 5 s-pause constants do not transfer to 30 s pauses, where idle time per turn dominates.\n")
    fl = M["flop"]["rows"]
    w("- FLOP model (J/FLOP fitted at 8k): " + "; ".join(
        f"nominal {x['L']//1000}k (actual {x['La']:,.0f}): actual-L error {sgn(x['ea'])}, nominal-L error {sgn(x['en'])}" for x in fl)
      + f". Measured 16k/4k ratio {fl[-1]['e']/fl[0]['e']:.3f} vs model {fl[-1]['pa']/fl[0]['pa']:.3f} (actual L) / "
      f"{flops_per_token(16000)/flops_per_token(4000):.3f} (nominal L). E2 marginal re-read at 64 users is under-predicted: "
      + ", ".join(f"{x['eng']} {x['L']:,.0f} tok {sgn(rel(x['pred'],x['marg']))}" for x in M["e2marg"] if x["u"] == 64) + ".\n")

    # Contradictions / not in plan
    w("## Where the data disagree with, or are missing from, the plan doc\n")
    o48 = [x for x in fa if x["lab"] == "SGLang 160 GB" and x["S"] == 48 and x["P"] == 30.0][0]
    w(f"1. The plan's refit says f = 0.76 predicts the other offload cells 'all within 0.5%', but it lists only three easy "
      f"cells. The same model gives {o48['pred']:.0f} vs {o48['meas']:.0f} J ({sgn(o48['err'])}) for 160 GB at 48 sessions, "
      f"30 s pause, which is not mentioned there.\n")
    w(f"2. H10's prompt lengths are labeled ~4k/8k/16k tokens; the measured prompts are "
      + "/".join(f"{x['La']:,.0f}" for x in fl) + " tokens (script `--lengths` is approximate). The plan's 'FLOP model 1.19x' "
      f"uses the nominal lengths; at the actual lengths the model gives {fl[-1]['pa']/fl[0]['pa']:.2f}x vs "
      f"{fl[-1]['e']/fl[0]['e']:.2f}x measured, so most of the 'extra' growth disappears.\n")
    w("3. H3's no-pressure clause: offload is ~5-6% cheaper per turn at 16 sessions on both engines, reproducibly (not 'within noise'); "
      "the plan never reports the 16-session offload cells.\n")
    gw = [x for v in fair.values() for x in v]
    w(f"4. Model (A) of H8 predicts no hits at 16 sessions/5 s (W/C {[x['wc'] for x in M['cliff']['points'] if x['S']==16 and x['P']==5.0][0]:.2f} "
      "> 1), but 97% of resumed turns hit there; W from final prompt sizes overstates the working set during the step, as the "
      "plan notes. Model B predicts full hits at 24 and 32 sessions/5 s (W/C 1.22, 1.55 <= ~1.55), where 29% and 0% of resumed "
      "turns hit. The plan's reading (collapse near 1.2 at 5 s, 1.9-2.2 at 30 s) matches the data; neither model fits.\n")
    w(f"5. Gate fairness, recomputed here: never served {min(x['never'] for x in gw)}-{max(x['never'] for x in gw)} sessions per step "
      f"across all gate cells (plan: 6-32 of 48); first-turn wait p90 up to {max(x['fw90'] for x in gw):.0f} s and max "
      f"{max(x['wmax'] for x in gw):.0f} s; recorded `gate_wait_p90` is {max(x['rec_w90'] or 0 for x in gw):.1f} s in every step. "
      f"Up to {max(x['yielded_end'] for x in gw)} sessions per step had given back their slot and were (or would have been) "
      "waiting again at the end, a second censored wait that is not recorded anywhere. The gate's p90 latency to first token "
      f"at 48/5 s with q=4 ({l90[('vllm-admit16x4',48,5.0)]:.0f} s) is worse than GPU-only's ({gref[(48,5.0)]:.0f} s), even "
      "before counting the sessions it never serves.\n")
    w("6. E2 and the seeds ran one round each, while the plan's E2 row and the reviewer table imply two-round repeatability; "
      "seed differences are between different sessions, not round noise.\n")
    w(f"7. SGLang's GPU KV capacity is {caps.get('sglang_gpu') or 'n/a':,} tokens, 2.7% more than vLLM's {C_GPU:,} that the "
      "plan uses for both engines; the cliff table uses each engine's own value.\n")
    w("8. H5 cannot be judged from CPU energy directly (no RAPL): the bound uses engine CPU time at 10 W/core and an assumed "
      "1 nJ/byte for host memory. The 160 GB run's CPU alignment is inferred from engine stops (no mtime log for it), "
      "validated on the 70 GiB and main runs (CPU-seconds within 3%). Two main-run vLLM GPU-only steps (round 1, 16/30 s and "
      "48/5 s) started before the CPU logger and one (16/5 s) was saved before it; the gate, LFU/SLRU, cliff and seed runs have "
      "no CPU alignment.\n")
    w(f"9. HotCarbon's 'throughput collapse at >= 64 GiB' is not seen for vLLM at 70 GiB: turns "
      f"{cell_mean(off,('vllm-offload70',48,5.0),'turns'):.0f} vs {cell_mean(m,('vllm-gpu',48,5.0),'turns'):.0f} (48/5 s), "
      f"{cell_mean(off,('vllm-offload70',96,5.0),'turns'):.0f} vs {cell_mean(m,('vllm-gpu',96,5.0),'turns'):.0f} (96/5 s). No failed turns in any run.\n")
    w("10. Results for H11-H18 quoted in the plan (controller, real trajectories, BFCL, TTL, MoE, sizing, write_back) are "
      "not in these zips and are unverified here.\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------------------
def main():
    extract()
    D = {}
    for fn in ("e2-len1600.json", "e2-len12600.json", "agents.json", "part2.json"):
        D[fn] = load(f"main/{fn}")
    for fn in ("agents-offload.json", "agents-admit.json", "agents-evict.json", "agents-cliff.json", "agents-host160.json",
               "transfer.json", "agents-seed-s1.json", "agents-seed-s2.json"):
        D[fn] = load(f"extra/{fn}")
    caps = capacities()

    tables, C = build_tables(D)
    with open(os.path.join(HERE, "tables.md"), "w") as fh:
        fh.write(tables)

    fair_txt, fair, ref = fairness(D, C)
    mod_txt, M = models(D, C, caps)
    cpu_txt, CPU, cpuinfo = cpu_section(D, C)
    cap_txt = ("## Capacities read from the server logs\n\n" + md_table(
        ["quantity", "tokens"], [[k, f(v)] for k, v in caps.items()]) +
        f"\nThe cliff model uses C = {C_GPU:,} (vLLM) as specified; SGLang points use its own value.\n")
    with open(os.path.join(HERE, "analysis.md"), "w") as fh:
        fh.write("# Analysis: fairness, model checks, CPU bound\n\nGenerated by `analyze.py`. Section 1 (tables) is "
                 "tables.md.\n\n" + fair_txt + "\n\n## 3. " + mod_txt.replace("# Model checks\n", "Model checks\n", 1)
                 .replace("\n## (", "\n### (") + "\n\n" + cpu_txt + "\n\n" + cap_txt)

    plt = setup_mpl()
    figs = [fig_cliff(plt, C, M), fig_ram(plt, C, caps), fig_transfer(plt, D, M), fig_gate(plt, C, fair, ref), fig_p2(plt, C)]
    plt.close("all")

    with open(os.path.join(HERE, "summary.md"), "w") as fh:
        fh.write(summary(D, C, M, fair, ref, CPU, caps, cpuinfo))
    print("wrote", ", ".join(["tables.md", "analysis.md", "summary.md"] + [os.path.relpath(p, HERE) for p in figs]))


if __name__ == "__main__":
    main()
