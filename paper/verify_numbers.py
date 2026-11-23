#!/usr/bin/env python3
"""Deterministic check of every quantitative claim in the paper against the raw result archives.

    cd research/kv-energy && python3 -I paper/verify_numbers.py

Python 3 standard library only. The result zips in ../results are read directly (nothing is extracted or executed).
Each check recomputes one number the paper prints (abstract, intro Findings, Sections 3-9, Tables 1-7, the outcome
column of Table 9, figure captions, discussion, conclusion) and compares it with the printed value at the printed
precision: a value printed with d decimals passes if the recomputation rounds to it (|x - printed| <= half a unit in
the last printed digit). "about X" values use X's significant figures. Ranges check both ends. "within/under/at most X"
claims pass if the recomputed value does not exceed X at X's precision. Every check also confirms that the printed
value still appears in the .tex source near the cited line; if it does not, the check fails.

Values the archives cannot give (citations, plan records, assumptions, hardware specs) are listed separately under
NOT FROM DATA; model calculations are recomputed from their stated inputs and checked like data.

Conventions (from make_figures.py / analysis/analyze.py and the run scripts):
  * per-step summary fields are used as stored in the archives (j_per_turn, cache_hit, j_per_prompt_token, ...);
  * "mean of rounds" = arithmetic mean of the per-round step values; "per-round range" = min..max over rounds;
  * percent change = 100 (a/b - 1), against the reference cell's two-round mean unless the rule says round-matched;
  * latency p90 of resumed turns = the stored resume_ttft_p90 / resume_latency_p90 (nearest rank, v[int(0.9 n)]);
    first-turn and all-turn p90 recomputed from turns_detail with v[int(0.9 n) - 1] (the definition that reproduces
    the plan's phase-9 table); gate wait is added to TTFT;
  * W (working set) = sum over sessions of each session's largest prompt in the step; C from the engine logs.
Exit code 1 if any check fails.
"""
import bisect
import csv
import datetime as dt
import io
import json
import math
import os
import re
import statistics
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
TEX = os.path.join(HERE, "tex")

# ======================================================================================================================
# Archive access
# ======================================================================================================================
_ZIPS, _JSON = {}, {}


def _zip(name):
    if name not in _ZIPS:
        _ZIPS[name] = zipfile.ZipFile(os.path.join(RES, name))
    return _ZIPS[name]


# The public archives omit engine logs and third-party data. For those members, fall back to the extracted startup-log
# lines in results/engine-log-lines/ and to the list of trajectory ids in code/swe_traj_ids.txt.
_FALLBACK = {"swe-traj-400.jsonl": os.path.join(HERE, "..", "code", "swe_traj_ids.txt")}


def raw(archive, member):
    z = _zip(archive)
    names = [n for n in z.namelist() if n == member or n.endswith("/" + member)]
    if names:
        return z.read(names[0])
    for alt in (os.path.join(RES, "engine-log-lines", member), _FALLBACK.get(member, "")):
        if alt and os.path.exists(alt):
            with open(alt, "rb") as fh:
                return fh.read()
    raise KeyError(f"{member} not in {archive}")


def J(archive, member):
    k = (archive, member)
    if k not in _JSON:
        _JSON[k] = json.loads(raw(archive, member))
    return _JSON[k]


def text(archive, member):
    return raw(archive, member).decode("utf-8", errors="ignore")


# ======================================================================================================================
# Data helpers
# ======================================================================================================================
def mean(v):
    v = list(v)
    return sum(v) / len(v)


def chg(a, b):
    """percent change of a against b"""
    return 100.0 * (a / b - 1.0)


def merge(*docs):
    return {"runs": [r for d in docs for r in d["runs"]]}


def sel(doc, var, N=None, pause=5.0, frac=None, rnd=None):
    out = []
    for r in doc["runs"]:
        if r["variant"] != var or (rnd is not None and r.get("round") != rnd):
            continue
        for s in r.get("steps", []):
            if N is not None and (s.get("sessions") != N or s.get("pause_median_s", pause) != pause):
                continue
            if frac is not None and s.get("fraction") != frac:
                continue
            out.append(s)
    assert out, (var, N, pause, frac, rnd)
    return out


def V(doc, var, key, N=None, pause=5.0, frac=None, rnd=None):
    return [s[key] for s in sel(doc, var, N, pause, frac, rnd)]


def M(doc, var, key, N=None, pause=5.0, frac=None, rnd=None):
    return mean(V(doc, var, key, N, pause, frac, rnd))


def rows(step):
    td = step["turns_detail"]
    ix = {k: i for i, k in enumerate(td["fields"])}
    return [{k: r[i] for k, i in ix.items()} for r in td["rows"]]


def p_nr(v, q=0.9):
    """nearest rank as in the run scripts (engine_energy2.pct)"""
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def p_lo(v, q=0.9):
    v = sorted(v)
    return v[max(0, int(q * len(v)) - 1)]


def served(step):
    return len({r["session"] for r in rows(step)})


def first_lat(step):
    return [r["ttft_s"] + r["gate_wait_s"] for r in rows(step) if r["turn"] == 0]


def all_lat(step):
    return [r["ttft_s"] + r["gate_wait_s"] for r in rows(step)]


def resumed_hit_share(step):
    tt = [t for t in rows(step) if t["turn"] > 0]
    return sum(1 for t in tt if t["cached_tokens"] >= 0.5 * t["prompt_tokens"]) / len(tt)


def working_set(step):
    mx = {}
    for t in rows(step):
        mx[t["session"]] = max(mx.get(t["session"], 0), t["prompt_tokens"])
    return sum(mx.values())


def log_int(archive, member, pat):
    m = re.search(pat, text(archive, member))
    return int(m.group(1).replace(",", ""))


# ======================================================================================================================
# Check machinery
# ======================================================================================================================
RESULTS = []
NFD = []
_TEXCACHE = {}


def _texlines(fn):
    if fn not in _TEXCACHE:
        with open(os.path.join(TEX, fn), encoding="utf-8") as fh:
            _TEXCACHE[fn] = fh.read().splitlines()
    return _TEXCACHE[fn]


def _norm(s):
    s = s.replace("{,}", ",").replace("--", "–").replace("\\%", "%").replace("$", "")
    s = s.replace("\\,", " ").replace("~", " ").replace("\\times", "x").replace("\\sim", "~")
    return s


def in_source(loc, printed):
    fn, ln = loc.split(":")
    ln = int(ln)
    L = _texlines(fn)
    win = _norm(" ".join(L[max(0, ln - 31): ln + 30]))  # +-30 lines: tolerant of edits above the cited line
    toks = re.findall(r"\d[\d,]*(?:\.\d+)?", printed)
    for t in toks:
        if not re.search(r"(?<![\d.,])" + re.escape(t) + r"(?![\d])", win):
            return False, t
    return True, None


def _num(s):
    s = s.strip().replace("−", "-").replace(",", "").replace("+", "").replace("%", "")
    return float(s)


def _unit(s, about=False):
    s = s.strip().replace("−", "").replace("-", "").replace("+", "").replace(",", "").replace("%", "")
    if "." in s:
        return 10.0 ** -len(s.split(".")[1])
    if about:  # significant figures: trailing zeros of an integer are not significant
        z = len(s) - len(s.rstrip("0"))
        return 10.0 ** min(z, len(s) - 1)
    return 1.0


def _split(printed):
    p = printed.replace("−", "-")
    if " to " in p:
        a, b = p.split(" to ")
    elif "–" in p:
        a, b = p.split("–")
    else:
        return [p]
    return [a, b]


def _fmt(x, u):
    nd = max(0, -int(math.floor(math.log10(u)))) + 2 if u < 1 else 2
    return f"{x:,.{nd}f}"


def C(label, loc, printed, comp, rule, kind="eq", unit=None, about=False, tol=0.0, note=""):
    """printed: the paper's string ('1,052', '56–58', '−10 to −9'); comp: number or (lo, hi)."""
    parts = _split(printed)
    if len(parts) == 2:
        pv = sorted([_num(parts[0]), _num(parts[1])])
        us = [unit or _unit(p, about) for p in parts]
        us = [x for _, x in sorted(zip([_num(parts[0]), _num(parts[1])], us))]
        cv = sorted(comp)
        ok = all(abs(c - p) <= u / 2 * (1 + 1e-9) + tol + 1e-12 for c, p, u in zip(cv, pv, us))
        shown = f"{_fmt(cv[0], us[0])} – {_fmt(cv[1], us[1])}"
    else:
        p, u = _num(parts[0]), unit or _unit(parts[0], about)
        c = comp
        if kind == "eq":
            ok = abs(c - p) <= u / 2 * (1 + 1e-9) + tol + 1e-12
        elif kind == "le":
            ok = c <= p + u / 2 * (1 + 1e-9) + tol
        elif kind == "ge":
            ok = c >= p - u / 2 * (1 + 1e-9) - tol
        else:
            raise ValueError(kind)
        shown = _fmt(c, u)
    src_ok, miss = in_source(loc, printed)
    if not src_ok:
        ok = False
        note = (note + "; " if note else "") + f"printed value {miss!r} not found near {loc}"
    RESULTS.append(dict(label=label, loc=loc, printed=printed, comp=shown, rule=rule, ok=ok, note=note))


def B(label, loc, printed, comp_desc, ok, rule, note=""):
    src_ok, miss = in_source(loc, printed)
    if not src_ok:
        ok = False
        note = (note + "; " if note else "") + f"printed value {miss!r} not found near {loc}"
    RESULTS.append(dict(label=label, loc=loc, printed=printed, comp=comp_desc, ok=bool(ok), rule=rule, note=note))


def N(text_, loc, category, note=""):
    NFD.append(dict(text=text_, loc=loc, cat=category, note=note))


# ======================================================================================================================
# Load archives
# ======================================================================================================================
A = J("all-results.zip", "agents.json")
P2 = J("all-results.zip", "part2.json")
E2S, E2L = J("all-results.zip", "e2-len1600.json"), J("all-results.zip", "e2-len12600.json")
ADMIT = J("extra-results.zip", "agents-admit.json")
CLIFF = J("extra-results.zip", "agents-cliff.json")
EVICT = J("extra-results.zip", "agents-evict.json")
H160 = J("extra-results.zip", "agents-host160.json")
OFF = J("extra-results.zip", "agents-offload.json")
SEED1, SEED2 = J("extra-results.zip", "agents-seed-s1.json"), J("extra-results.zip", "agents-seed-s2.json")
TR = J("extra-results.zip", "transfer.json")
CTRL = J("phase2-results.zip", "agents-ctrl.json")
LONG = J("phase2-results.zip", "agents-long.json")
REVICT = J("phase2-results.zip", "agents-real-evict.json")
REAL = J("phase2-results.zip", "agents-real.json")
RET = J("phase2-results.zip", "retention.json")
MOEREAL = J("phase3-results.zip", "agents-moe-real.json")
MOE = J("phase3-results.zip", "agents-moe.json")
REALHI = J("phase3-results.zip", "agents-real-hi.json")
SIZING = J("phase3-results.zip", "agents-sizing.json")
IDLE = J("phase3-results.zip", "idle-cache.json")
FSRR = J("phase4-results.zip", "agents-fullstack-rerun.json")
MC1, MC2 = J("phase6-results.zip", "mooncake-r1.json"), J("phase6-results.zip", "mooncake-r2.json")
F8B1 = J("phase7-results.zip", "agents-p5-8b-formula-r1.json")
BFCL = J("phase7-results.zip", "bfcl.json")
MCW2 = J("phase7-results.zip", "mooncake-w2.json")
AD8 = J("phase8-results.zip", "agents-8b-adaptive.json")
LC1, LC2 = J("phase8-results.zip", "agents-longctx-r1.json"), J("phase8-results.zip", "agents-longctx-r2.json")
LC = merge(LC1, LC2)
P98B, P9LC = J("phase9-results.zip", "agents-p9-8b.json"), J("phase9-results.zip", "agents-p9-lc.json")
P58B = J("rest-results.zip", "agents-p5-8b.json")
B8 = merge(P58B, F8B1)
CHEAP = J("rest-results.zip", "agents-p5-cheap.json")
FAIR = J("rest-results.zip", "agents-p5-fair.json")
R3 = J("rest-results.zip", "agents-p5-r3.json")
P5REAL = J("rest-results.zip", "agents-p5-real.json")

# Capacities from the engines' startup logs
CAP = dict(
    vllm32=log_int("all-results.zip", "vllm-gpu-agent-r1-server.log", r"GPU KV cache size: ([\d,]+) tokens"),
    sgl32=log_int("all-results.zip", "sglang-gpu-agent-r1-server.log", r"max_total_num_tokens=(\d+)"),
    moe=log_int("rest-results.zip", "vllm-gpu-moe-agent-r1-server.log", r"GPU KV cache size: ([\d,]+) tokens"),
    vllm8=log_int("rest-results.zip", "vllm-gpu-8b-agent-r1-server.log", r"GPU KV cache size: ([\d,]+) tokens"),
    sgl8=log_int("rest-results.zip", "sglang-gpu-8b-agent-r1-server.log", r"max_total_num_tokens=(\d+)"),
    host80=log_int("all-results.zip", "sglang-offload-agent-r1-server.log", r"KV host pool: (\d+) tokens"),
    host160=log_int("extra-results.zip", "sglang-offload160-agent-r1-server.log", r"KV host pool: (\d+) tokens"),
)
WEIGHTS_GIB = float(re.search(r"Model loading took ([\d.]+) GiB", text("all-results.zip", "vllm-gpu-agent-r1-server.log")).group(1))
CV = CAP["vllm32"]
KV32 = 2 * 64 * 8 * 128 * 2  # bytes per token, Qwen3-32B BF16

# ======================================================================================================================
# Derived quantities used in several places
# ======================================================================================================================
# --- thrash = no cache: |delta| per overloaded cell and round
thr_vllm, thr_sgl, thr_all = [], [], []
for N_ in (48, 96):
    for P in (5.0, 30.0):
        for rd in (1, 2):
            thr_vllm.append(abs(chg(sel(A, "vllm-gpu", N_, P, rnd=rd)[0]["j_per_turn"], sel(A, "vllm-nocache", N_, P, rnd=rd)[0]["j_per_turn"])))
            g, n = sel(A, "sglang-gpu", N_, P, rnd=rd)[0], sel(A, "sglang-nocache", N_, P, rnd=rd)[0]
            thr_sgl.append(abs(chg(g["j_per_turn"], n["j_per_turn"])))
sgl_same_turns = all(sel(A, "sglang-gpu", N_, P, rnd=rd)[0]["turns"] == sel(A, "sglang-nocache", N_, P, rnd=rd)[0]["turns"]
                     for N_ in (48, 96) for P in (5.0, 30.0) for rd in (1, 2))
thr_r3 = [abs(chg(M(R3, "vllm-gpu-r3", "j_per_turn", N_), M(R3, "vllm-nocache-r3", "j_per_turn", N_))) for N_ in (48, 96)]
thr_moe = [abs(chg(a, b)) for N_ in (96, 192) for a, b in zip(V(MOE, "vllm-gpu-moe", "j_per_turn", N_), V(MOE, "vllm-nocache-moe", "j_per_turn", N_))]
thr_8b = [abs(chg(a, b)) for N_ in (128, 256) for a, b in zip(V(P58B, "vllm-gpu-8b", "j_per_turn", N_), V(P58B, "vllm-nocache-8b", "j_per_turn", N_))]
thr_lc = [abs(chg(a, M(LC, "vllm-nocache-lc", "j_per_turn", N_))) for N_ in (16, 32) for a in V(LC, "vllm-gpu-lc", "j_per_turn", N_)]
thr_all = thr_vllm + thr_sgl + thr_r3 + thr_moe + thr_8b + thr_lc
thr_seeds = [abs(chg(M(S_, "vllm-gpu", "j_per_turn", N_, P), M(S_, "vllm-nocache", "j_per_turn", N_, P)))
             for S_ in (SEED1, SEED2) for N_ in (48,) for P in (5.0, 30.0)]

# --- cliff: hit against W/C (vLLM, 11k sessions), as analysis/analyze.py (d)
cliff_pts = []
for doc, Ns in ((A, (16, 48, 96)), (CLIFF, (24, 32, 40))):
    for N_ in Ns:
        for P in (5.0, 30.0):
            for s in sel(doc, "vllm-gpu", N_, P):
                cliff_pts.append(dict(N=N_, P=P, wc=working_set(s) / CV, hit=s["cache_hit"], rh=resumed_hit_share(s)))
cross = {}
for P in (5.0, 30.0):
    sat = mean(x["rh"] for x in cliff_pts if x["N"] == 16 and x["P"] == P)
    pts = sorted({(round(x["wc"], 4), x["rh"]) for x in cliff_pts if x["P"] == P})
    c = None
    for (a, ya), (b, yb) in zip(pts, pts[1:]):
        if ya >= sat / 2 > yb:
            c = a + (ya - sat / 2) / (ya - yb) * (b - a)
    last = max(a for a, y in pts if y > 0.02)
    first0 = min(a for a, y in pts if y <= 0.02 and a > last)
    cross[P] = (c, last, first0)
WC_40_30 = mean(x["wc"] for x in cliff_pts if x["N"] == 40 and x["P"] == 30.0)
W48 = mean(working_set(s) for s in sel(A, "vllm-gpu", 48, 5.0))


def hit_cliff(N_, P):
    doc = A if N_ in (16, 48, 96) else CLIFF
    return 100 * M(doc, "vllm-gpu", "cache_hit", N_, P)


# --- transfer (Table 1)
def tr_phase(variant, phase, length):
    return [s[phase] for r in TR["runs"] if r["variant"] == variant for s in r["steps"] if s["length"] == length]


LENS = (4000, 8000, 16000)
rec6 = {L: [x["above_idle_j_per_prompt_token"] * 1e3 for v in ("sglang-gpu-control", "sglang-host160") for x in tr_phase(v, "recompute", L)] for L in LENS}
rec9 = {L: rec6[L] + [x["above_idle_j_per_prompt_token"] * 1e3 for x in tr_phase("sglang-gpu-control", "after_flush", L)] for L in LENS}
rst = {L: [x["above_idle_j_per_prompt_token"] * 1e3 for x in tr_phase("sglang-host160", "after_flush", L)] for L in LENS}
hit_tr = [x["above_idle_j_per_prompt_token"] * 1e3 for v in ("sglang-gpu-control", "sglang-host160") for L in LENS for x in tr_phase(v, "gpu_hit", L)]
ratio_mean = {L: mean(rec9[L]) / mean(rst[L]) for L in LENS}
ratio_rep = [mean(rec9[L]) / r for L in LENS for r in rst[L]]
Lact = {L: mean(x["prompt_tokens"] / x["requests"] for v in ("sglang-gpu-control", "sglang-host160") for x in tr_phase(v, "recompute", L)) for L in LENS}


def F(L):  # Eq. 1, FLOP per prompt token
    return 62.4e9 + 2.10e6 * L / 2


# ======================================================================================================================
# CPU-time bound for host tiers (port of analysis/analyze.py, section 4)
# ======================================================================================================================
COOLDOWN_S, W_PER_CORE, J_PER_BYTE = 20.0, 10.0, 1e-9


def load_cpu(member):
    T, Cc, Mt = [], [], []
    for r in csv.DictReader(io.StringIO(text("extra-results.zip", member))):
        T.append(float(r["unix_s"])); Cc.append(float(r["engine_cpu_s"]))
        Mt.append(float(r["agents_json_mtime"]) if r["agents_json_mtime"] else None)
    return T, Cc, Mt


def window_cpu(T, Cc, t0, t1):
    if t0 < T[0] or t1 > T[-1]:
        return None
    i0, i1 = bisect.bisect_left(T, t0), bisect.bisect_right(T, t1) - 1
    tot = 0.0
    for i in range(max(i0, 1), min(i1 + 2, len(T))):
        if Cc[i - 1] > 50 and Cc[i] < 0.5 * Cc[i - 1]:
            return None
        inc = max(0.0, Cc[i] - Cc[i - 1])
        a, b = T[i - 1], T[i]
        lo, hi = max(a, t0), min(b, t1)
        if hi > lo:
            tot += inc * (hi - lo) / (b - a)
    return tot


def save_events(Mt, merge_s=5.0):
    ev, last = [], None
    for m in Mt:
        if m is None or m == last:
            continue
        last = m
        if ev and m - ev[-1]["t"] < merge_s:
            ev[-1]["t"] = m; ev[-1]["n"] += 1
        else:
            ev.append({"t": m, "n": 1})
    return ev


def align(doc, ev):
    best = None
    for wf in (False, True):
        X = [("step", ri, k) for ri, r in enumerate(doc["runs"]) for k in range(len(r["steps"]))]
        X = []
        for ri, r in enumerate(doc["runs"]):
            X += [("step", ri, k) for k in range(len(r["steps"]))]
            X.append(("end", ri, None))
        if wf:
            X.append(("final", None, None))
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
        cand = (viol, -n, pairs)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return best[2]


def stops(T, Cc, t0, t1, minc=200.0):
    return [T[i] for i in range(1, len(T)) if t0 <= T[i] <= t1 and Cc[i - 1] > minc and Cc[i] < 0.5 * Cc[i - 1]]


def _rec(run, s, t1, cpu, cpu_hi):
    return dict(variant=run["variant"], round=run["round"], S=s["sessions"], P=s["pause_median_s"], turns=s["turns"],
                wall=s["wall_s"], t_save=t1, cpu=cpu, cpu_hi=cpu_hi, cached=s["cached_tokens"], recomputed=s["recomputed_tokens"])


def do_aligned(doc, T, Cc, Mt):
    pairs = align(doc, save_events(Mt))
    res, ovh = [], []
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
        res.append(_rec(run, s, t1, window_cpu(T, Cc, t0, t1), window_cpu(T, Cc, t0 - max(o or 0, 0) - 2, t1)))
    return res, ovh


def _ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def host_cost():
    Tm, Cm, Mm = load_cpu("cpu-time.csv")
    To, Co, Mo = load_cpu("cpu-time-offload.csv")
    main_res, ov1 = do_aligned(A, Tm, Cm, Mm)
    off_res, ov2 = do_aligned(OFF, To, Co, Mo)
    lag = {"vllm": [], "sglang": []}
    for doc, T, Cc, res in ((A, Tm, Cm, main_res), (OFF, To, Co, off_res)):
        for r in doc["runs"]:
            last = [x for x in res if x["variant"] == r["variant"] and x["round"] == r["round"]]
            if len(last) == len(r["steps"]) and last:
                ls = max(x["t_save"] for x in last)
                st = stops(T, Cc, ls, ls + 300)
                if st:
                    lag[r["engine"]].append(st[0] - ls)
    g_med = statistics.median([o for o in ov1 + ov2 if o is not None])
    # inferred alignment for the 160 GB run (no mtime log): back from each engine stop
    res = []
    st = stops(Tm, Cm, _ts(H160["started_at"]), _ts(H160["finished_at"]) + 120)
    runs = [r for r in H160["runs"] if r["steps"]]
    assert len(st) == len(runs), "160 GB run: engine stops do not match runs"
    for r, t_stop in zip(runs, st):
        t = t_stop - statistics.median(lag[r["engine"]])
        saves = []
        for k in range(len(r["steps"]) - 1, -1, -1):
            saves.append((k, t))
            t = t - r["steps"][k]["wall_s"] - COOLDOWN_S - g_med
        for k, t1 in sorted(saves):
            s = r["steps"][k]
            res.append(_rec(r, s, t1, window_cpu(Tm, Cm, t1 - s["wall_s"], t1), window_cpu(Tm, Cm, t1 - s["wall_s"] - 5, t1 + 5)))
    allres = main_res + off_res + res

    def cpu_per_turn(variant, S, P):
        xs = [x for x in allres if x["variant"] == variant and x["S"] == S and x["P"] == P and x["cpu"] is not None]
        return sum(x["cpu"] for x in xs) / sum(x["turns"] for x in xs) if xs else None

    out = {}
    for ov, gv, doc, lab in (("vllm-offload", "vllm-gpu", OFF, "vllm70"), ("sglang-offload", "sglang-gpu", A, "sgl80"),
                             ("sglang-offload160", "sglang-gpu", H160, "sgl160")):
        for S in (16, 48, 96):
            for P in (5.0, 30.0):
                co, cg = cpu_per_turn(ov, S, P), cpu_per_turn(gv, S, P)
                saved = M(A, gv, "j_per_turn", S, P) - M(doc, ov, "j_per_turn", S, P)
                xs = [x for x in allres if x["variant"] == ov and x["S"] == S and x["P"] == P]
                mem_j = J_PER_BYTE * mean((x["recomputed"] + x["cached"]) * KV32 / x["turns"] for x in xs)
                if co is None or cg is None or saved <= 0:
                    out[(lab, S, P)] = None
                    continue
                out[(lab, S, P)] = 100 * (max(W_PER_CORE * (co - cg), 0) + mem_j) / saved
    return out


HOST = host_cost()

# ======================================================================================================================
# Shorthands for frequently used cells
# ======================================================================================================================
G32 = {(N_, P): M(A, "vllm-gpu", "j_per_turn", N_, P) for N_ in (16, 48, 96) for P in (5.0, 30.0)}
SG32 = {(N_, P): M(A, "sglang-gpu", "j_per_turn", N_, P) for N_ in (16, 48, 96) for P in (5.0, 30.0)}
SGH = {(N_, P): M(A, "sglang-gpu", "cache_hit", N_, P) for N_ in (16, 48, 96) for P in (5.0, 30.0)}
VGH = {(N_, P): M(A, "vllm-gpu", "cache_hit", N_, P) for N_ in (16, 48, 96) for P in (5.0, 30.0)}
F_FIT = (1 - M(H160, "sglang-offload160", "j_per_turn", 48) / SG32[(48, 5.0)]) / (M(H160, "sglang-offload160", "cache_hit", 48) - SGH[(48, 5.0)])


def pred_tier(doc, var, N_, P, base="sglang"):
    Et, ht = (SG32[(N_, P)], SGH[(N_, P)]) if base == "sglang" else (G32[(N_, P)], VGH[(N_, P)])
    return Et * (1 - F_FIT * (M(doc, var, "cache_hit", N_, P) - ht)), M(doc, var, "j_per_turn", N_, P)


def mc(doc, var, frac=0.25, rnd=None):
    return sel(doc, var, frac=frac, rnd=rnd)[0]


W1 = {v: mc(MC1, v) for v in ("vllm-nocache-mc", "vllm-gpu-mc", "sglang-gpu-mc", "sglang-host160-mc", "sglang-host300-mc")}
W1R2 = {v: mc(MC2, v) for v in ("vllm-nocache-mc", "vllm-gpu-mc", "sglang-host160-mc")}
W2 = {v: mc(MCW2, v) for v in ("vllm-nocache-mc", "vllm-gpu-mc", "sglang-host160-mc")}
W2E = {v: mc(MCW2, v, 0.125) for v in ("vllm-nocache-mc", "vllm-gpu-mc", "sglang-host160-mc")}
mjt = lambda s: 1000 * s["j_per_prompt_token"]
done = lambda s: 100 * s["completed"] / s["requests"]
PRED1 = MC1["samples"]["0.25"]["lru_hit_pred"]
PRED2Q, PRED2E = MCW2["samples"]["0.25"]["lru_hit_pred"], MCW2["samples"]["0.125"]["lru_hit_pred"]
PLAN_PRED_300 = 0.525  # plan record (300 GB block-LRU prediction is not in the archives)
F_P = 0.76             # f as printed and used for production predictions

ratio = lambda s, n: s["j_per_prompt_token"] / n["j_per_prompt_token"]
prod_model = [  # (pred, measured) on the busy card
    (1 - F_P * W1["vllm-gpu-mc"]["cache_hit"], ratio(W1["vllm-gpu-mc"], W1["vllm-nocache-mc"])),
    (1 - F_P * W1R2["vllm-gpu-mc"]["cache_hit"], ratio(W1R2["vllm-gpu-mc"], W1R2["vllm-nocache-mc"])),
    (1 - F_P * W1["sglang-host160-mc"]["cache_hit"], ratio(W1["sglang-host160-mc"], W1["vllm-nocache-mc"])),
    (1 - F_P * W1["sglang-host300-mc"]["cache_hit"], ratio(W1["sglang-host300-mc"], W1["vllm-nocache-mc"])),
    (1 - F_P * W2["vllm-gpu-mc"]["cache_hit"], ratio(W2["vllm-gpu-mc"], W2["vllm-nocache-mc"])),
    (1 - F_P * W2["sglang-host160-mc"]["cache_hit"], ratio(W2["sglang-host160-mc"], W2["vllm-nocache-mc"])),
]
prod_model_err = [abs(100 * (p / m - 1)) for p, m in prod_model]
keepup_hit_err = [abs(100 * W2E["vllm-gpu-mc"]["cache_hit"] - 100 * PRED2E["244928"]),
                  abs(100 * W2E["sglang-host160-mc"]["cache_hit"] - 100 * PRED2E["610351"]),
                  abs(100 * W1["sglang-host160-mc"]["cache_hit"] - 100 * PRED1["610351"]),
                  abs(100 * W1["sglang-host300-mc"]["cache_hit"] - 100 * PLAN_PRED_300),
                  abs(100 * W2["sglang-host160-mc"]["cache_hit"] - 100 * PRED2Q["610351"])]
chain = [100 * ((1 - F_P * PRED1["610351"]) / ratio(W1["sglang-host160-mc"], W1["vllm-nocache-mc"]) - 1),
         100 * ((1 - F_P * PLAN_PRED_300) / ratio(W1["sglang-host300-mc"], W1["vllm-nocache-mc"]) - 1)]
tier_more_tok = [chg(W1["sglang-host160-mc"]["prompt_tokens"], W1["vllm-gpu-mc"]["prompt_tokens"]),
                 chg(W2["sglang-host160-mc"]["prompt_tokens"], W2["vllm-gpu-mc"]["prompt_tokens"])]
tier_less_mj = [-chg(mjt(W1["sglang-host160-mc"]), mjt(W1["vllm-gpu-mc"])), -chg(mjt(W1["sglang-host160-mc"]), mjt(W1["sglang-gpu-mc"])),
                -chg(mjt(W2["sglang-host160-mc"]), mjt(W2["vllm-gpu-mc"]))]
gpu_done = [done(W1["vllm-gpu-mc"]), done(W1["sglang-gpu-mc"]), done(W2["vllm-gpu-mc"])]

# adaptive gate (Table 6) and 10-minute steps (Table 7)
AD = {}
for lab, Rg, g, Ra, a, Ns in (("8b", P58B, "vllm-gpu-8b", AD8, "vllm-adaptive-8b", (64, 128, 256)),
                              ("lc", LC, "vllm-gpu-lc", LC, "vllm-adaptive-lc", (4, 8, 16, 32))):
    for N_ in Ns:
        AD[(lab, N_)] = dict(gJ=M(Rg, g, "j_per_turn", N_), aJ=M(Ra, a, "j_per_turn", N_),
                             gp=V(Rg, g, "resume_ttft_p90", N_), ap=V(Ra, a, "resume_latency_p90", N_),
                             gs=sel(Rg, g, N_), as_=sel(Ra, a, N_))
T10 = {k: sel(d, v, n)[0] for k, d, v, n in (("8b-gpu", P98B, "vllm-gpu-8b-10m", 256), ("8b-gate", P98B, "vllm-adaptive-8b-10m", 256),
                                             ("lc-gpu", P9LC, "vllm-gpu-lc-10m", 32), ("lc-gate", P9LC, "vllm-adaptive-lc-10m", 32))}


def min_turns(step):
    c = {}
    for r in rows(step):
        c[r["session"]] = c.get(r["session"], 0) + 1
    return min(c.values()), c


# BFCL
BF = {r["variant"]: r for r in BFCL["runs"] if r.get("energy_j")}


def bfcl_score(var, cat):
    t = text("phase7-results.zip", f"bfcl-scores/{var}-r1/Qwen_Qwen3-32B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_score.json")
    lines = t.splitlines()
    head = json.loads(lines[0])
    errs = 0
    for l in lines[1:]:
        e = json.loads(l)
        et = e.get("error", {}).get("error_type") if isinstance(e.get("error"), dict) else e.get("error_type")
        errs += et == "multi_turn:inference_error"
    return head["correct_count"], head["total_count"], errs


BS = {v: {c: bfcl_score(v, c) for c in ("base", "long_context")} for v in ("vllm-gpu-bfcl", "vllm-nocache-bfcl")}
corr = {v: sum(BS[v][c][0] for c in BS[v]) for v in BS}
kwh = {v: BF[v]["energy_j"] / 3.6e6 for v in BF}
perkwh = {v: corr[v] / kwh[v] for v in BF}

# ======================================================================================================================
# ABSTRACT
# ======================================================================================================================
f = "abstract.tex"
C("cliff W/C band (5 s half-hit .. 30 s vanish)", f + ":8", "1.2–2.2", (cross[5.0][0], cross[30.0][2]),
  "W/C where resumed-turn hits fall to half (5 s, interpolated) and first W/C with ~0 hits (30 s); vLLM, all rounds")
C("thrash vs no cache, |dJ/turn| over all overloaded cells", f + ":9", "0.01–1.6", (min(thr_all), max(thr_all)),
  "per-round |GPU/no-cache - 1| in J/turn: 32B vLLM+SGLang 48/96 x 5/30 s (r1,r2), 32B r3, 30B-A3B, 8B, 27k")
B("5 s: collapse between 16 and 32 sessions (~11k)", f + ":10", "16 and 32",
  f"hit {hit_cliff(16, 5.0):.1f}% at 16, {hit_cliff(32, 5.0):.2f}% at 32", hit_cliff(16, 5.0) > 50 and hit_cliff(32, 5.0) < 1,
  "vLLM 11k sessions, mean of rounds: hit > 50% at the lower bound and < 1% at the upper")
med11 = statistics.median([r["prompt_tokens"] for N_ in (16, 48, 96) for s in sel(A, "vllm-gpu", N_) for r in rows(s)])
C("session size 'about 11k'", f + ":10", "11", med11 / 1000, "median prompt over all requests of the vLLM 11k GPU-cache steps (5 s, 16-96), k tokens", about=True)
B("27k: collapse between 8 and 16 sessions", f + ":11", "8 and 16",
  f"hit {100*M(LC,'vllm-gpu-lc','cache_hit',8):.1f}% at 8, {100*M(LC,'vllm-gpu-lc','cache_hit',16):.2f}% at 16",
  M(LC, "vllm-gpu-lc", "cache_hit", 8) > 0.5 and M(LC, "vllm-gpu-lc", "cache_hit", 16) < 0.01, "long-context vLLM, mean of rounds")
med27 = statistics.median([r["prompt_tokens"] for N_ in (4, 8, 16, 32) for s in sel(LC, "vllm-gpu-lc", N_) for r in rows(s)])
C("session size 'about 27k'", f + ":11", "27", med27 / 1000, "median prompt over all requests of the long-context GPU-cache steps, k tokens", about=True)
C("restore/recompute 1/110", f + ":11", "110", ratio_mean[4000], "ratio of means (recompute n=9 / restore n=3), ~5k prompts", about=True)
C("restore/recompute 1/220", f + ":12", "220", ratio_mean[16000], "ratio of means, ~20k prompts", about=True)
B("160 GB tier finished every request (1/4, both windows)", f + ":13", "1/4", f"{done(W1['sglang-host160-mc']):.0f}% and {done(W2['sglang-host160-mc']):.0f}%",
  done(W1["sglang-host160-mc"]) == 100 and done(W2["sglang-host160-mc"]) == 100 and done(W1R2["sglang-host160-mc"]) == 100, "completed/requests, all rounds")
C("GPU cache finished 85-86%", f + ":13", "85–86", (min(gpu_done), max(gpu_done)), "completed/requests, GPU cache (vLLM w1, SGLang w1, vLLM w2), 1/4")
C("16-18% more prompt tokens", f + ":13", "16–18", (min(tier_more_tok), max(tier_more_tok)), "finished prompt tokens, 160 GB vs vLLM GPU cache, windows 1 and 2")
C("1/8 sample: tier saved under 1%", f + ":14", "1", -chg(mjt(W2E["sglang-host160-mc"]), mjt(W2E["vllm-gpu-mc"])), "J per prompt token, 160 GB vs GPU, w2 1/8", kind="le")
C("one-parameter model predicts busy card to about 1%", f + ":14", "1", max(prod_model_err), "max |pred/meas-1| over busy production cells (w1 GPU r1/r2, 160, 300; w2 GPU, 160), f=0.76", kind="le", about=True)
gate_trace_ok = all(x["gate_on"] == (x["w"] > 1.2 * s["controller"]["capacity"]) for s in sel(AD8, "vllm-adaptive-8b") for x in s["controller"]["trace"])
B("gate switches on when working set > 1.2 C", f + ":16", "1.2", "controller trace gate_on == (w > 1.2 C) in every interval" if gate_trace_ok else "mismatch",
  gate_trace_ok, "adaptive-gate controller traces, Qwen3-8B, both rounds")
below_ok = all(AD[k]["aJ"] <= AD[k]["gJ"] for k in (("8b", 64), ("lc", 4), ("lc", 8)))
B("gate cost no energy below the cliff", f + ":16", "1.2", "adaptive <= GPU J/turn at 8B/64, 27k/4, 27k/8" if below_ok else "adaptive higher", below_ok, "mean of rounds")
more_turns10 = [chg(T10["8b-gate"]["turns"], T10["8b-gpu"]["turns"]), chg(T10["lc-gate"]["turns"], T10["lc-gpu"]["turns"])]
less_e10 = [-chg(T10["8b-gate"]["j_per_turn"], T10["8b-gpu"]["j_per_turn"]), -chg(T10["lc-gate"]["j_per_turn"], T10["lc-gpu"]["j_per_turn"])]
all_served10 = served(T10["8b-gate"]) == 256 and served(T10["lc-gate"]) == 32
B("10-min: gate served every session", f + ":17", "48–62", f"{served(T10['8b-gate'])}/256, {served(T10['lc-gate'])}/32", all_served10, "sessions with >= 1 turn")
C("10-min: 48-62% more turns", f + ":17", "48–62", (min(more_turns10), max(more_turns10)), "turns, gate vs GPU cache, one round")
C("10-min: 42-50% less energy per turn", f + ":17", "42–50", (min(less_e10), max(less_e10)), "J/turn, gate vs GPU cache, one round")
C("BFCL accuracy changed by 3 of 400", f + ":18", "3", corr["vllm-gpu-bfcl"] - corr["vllm-nocache-bfcl"], "correct (base + long-context), one round")
B("BFCL 'doubled' correct answers per kWh", f + ":18", "400", f"{perkwh['vllm-gpu-bfcl']/perkwh['vllm-nocache-bfcl']:.2f}x",
  round(perkwh["vllm-gpu-bfcl"] / perkwh["vllm-nocache-bfcl"], 1) == 2.0, "ratio of correct answers per kWh rounds to 2.0")

# ======================================================================================================================
# INTRODUCTION (Findings)
# ======================================================================================================================
f = "intro.tex"
C("recompute 56-73 mJ per prompt token", f + ":6", "56–73", (min(rec6[4000]), max(rec6[16000])), "Table 1 recompute range, min at ~5k, max at ~20k (mJ, above idle)")
C("cliff factor 1.2-2.2", f + ":11", "1.2–2.2", (cross[5.0][0], cross[30.0][2]), "as abstract")
C("thrash = no cache within 0.01-1.6%", f + ":35", "0.01–1.6", (min(thr_all), max(thr_all)), "as abstract (two engines, three models)")
C("5 s: hits half at W/C", f + ":36", "1.2", cross[5.0][0], "interpolated W/C where resumed turns that hit fall below half the 16-session level")
C("5 s: hits vanish by W/C", f + ":36", "1.55", cross[5.0][2], "first W/C with <= 2% resumed turns hitting")
C("30 s: hits half at W/C", f + ":36", "1.5", cross[30.0][0], "interpolated")
C("30 s: hits vanish by W/C", f + ":37", "2.2", cross[30.0][2], "first W/C at ~0")
C("1/s between sessions of about 11k", f + ":37", "11", med11 / 1000, "as abstract", about=True)
C("... and about 27k tokens", f + ":37", "27", med27 / 1000, "as abstract", about=True)
idle_d = {r["variant"]: [100 * (p["idle_w"] / r["phases"][0]["idle_w"] - 1) for p in r["phases"][1:]] for r in IDLE["runs"]}
C("full cache changed idle power by < 0.2%", f + ":38", "0.2", max(abs(x) for v in idle_d.values() for x in v), "max |idle(full)/idle(empty)-1|, both engines, two readings", kind="le")
C("restore 1/110 (intro)", f + ":39", "110", ratio_mean[4000], "ratio of means", about=True)
C("restore 1/220 (intro)", f + ":39", "220", ratio_mean[16000], "ratio of means", about=True)
C("GPU cache alone finished 85-86%", f + ":42", "85–86", (min(gpu_done), max(gpu_done)), "as abstract")
C("16-18% more prompt tokens (intro)", f + ":42", "16–18", (min(tier_more_tok), max(tier_more_tok)), "as abstract")
C("15-17% less energy per prompt token", f + ":43", "15–17", (min(tier_less_mj), max(tier_less_mj)), "160 GB vs GPU cache J/prompt token: w1 vs vLLM, w1 vs SGLang, w2 vs vLLM")
C("1/8 sample saved under 1% (intro)", f + ":44", "1", -chg(mjt(W2E["sglang-host160-mc"]), mjt(W2E["vllm-gpu-mc"])), "as abstract", kind="le")
C("model within about 1% on busy card", f + ":48", "1", max(prod_model_err), "as abstract", kind="le", about=True)
C("hit predictor within 0.04-4.3 pp when card keeps up", f + ":48", "0.04–4.3", (min(keepup_hit_err), max(keepup_hit_err)),
  "|measured - block-LRU predicted| hit, cells that finished every request (w2 1/8 GPU+160; w1 160, 300; w2 1/4 160)", tol=0.05,
  note="300 GB prediction 0.525 is a plan record (3 decimals): +-0.05 pp input tolerance")
below_p90_ok = all(mean(AD[k]["ap"]) <= mean(AD[k]["gp"]) for k in (("8b", 64), ("lc", 4), ("lc", 8)))
B("below cliff resumed-turn p90 no worse", f + ":51", "90", "adaptive mean p90 <= GPU mean p90 at 8B/64, 27k/4, 27k/8" if below_p90_ok else "worse",
  below_p90_ok, "mean of rounds of resume_latency_p90 (gate) vs resume_ttft_p90 (GPU)")
ad_save = [-chg(AD[k]["aJ"], AD[k]["gJ"]) for k in (("8b", 128), ("8b", 256), ("lc", 16), ("lc", 32))]
C("150 s steps: gate cut energy 41-45%", f + ":52", "41–45", (min(ad_save), max(ad_save)), "mean of rounds, cells above the cliff (8B 128/256, 27k 16/32)")
unserved = [1 - served(s) / 256 for s in AD[("8b", 256)]["as_"]] + [1 - served(s) / 32 for s in AD[("lc", 32)]["as_"]]
B("a third to a half of sessions unserved at highest loads", f + ":52", "150",
  "unserved share " + ", ".join(f"{100*x:.0f}%" for x in unserved), all(0.30 <= x <= 0.5 + 1e-9 for x in unserved),
  "1 - served/N at 8B/256 and 27k/32, both rounds; 'a third'..'a half' read as 30-50%")
B("600 s steps: served every session", f + ":53", "600", f"{served(T10['8b-gate'])}/256, {served(T10['lc-gate'])}/32", all_served10, "phase 9")
C("600 s: 48-62% more turns", f + ":53", "48–62", (min(more_turns10), max(more_turns10)), "as abstract")
C("600 s: 42-50% less energy", f + ":53", "42–50", (min(less_e10), max(less_e10)), "as abstract")
fp = {k: p_lo(first_lat(T10[k])) for k in T10}
r10 = [fp["8b-gate"] / fp["8b-gpu"], fp["lc-gate"] / fp["lc-gpu"]]
C("600 s first-turn p90 1.9-2.7x GPU", f + ":54", "1.9–2.7", (min(r10), max(r10)), "ratio of first-turn p90 (TTFT + gate wait), gate / GPU")
B("BFCL 3 of 400", f + ":55", "3 of 400", f"{corr['vllm-gpu-bfcl']} vs {corr['vllm-nocache-bfcl']} of {sum(BS['vllm-gpu-bfcl'][c][1] for c in BS['vllm-gpu-bfcl'])}",
  corr["vllm-gpu-bfcl"] - corr["vllm-nocache-bfcl"] == 3 and sum(BS["vllm-gpu-bfcl"][c][1] for c in BS["vllm-gpu-bfcl"]) == 400, "score files")
C("BFCL correct with cache", f + ":55", "204", corr["vllm-gpu-bfcl"], "score files, base + long-context")
C("BFCL correct without cache", f + ":56", "201", corr["vllm-nocache-bfcl"], "score files")
C("BFCL 2.0x correct per kWh", f + ":56", "2.0", perkwh["vllm-gpu-bfcl"] / perkwh["vllm-nocache-bfcl"], "ratio of correct/kWh")
B("400 SWE-agent trajectories", f + ":23", "400", f"{len(text('phase2-results.zip', 'swe-traj-400.jsonl').splitlines())} lines",
  len(text("phase2-results.zip", "swe-traj-400.jsonl").splitlines()) == 400, "swe-traj-400.jsonl in phase2-results.zip")

# ======================================================================================================================
# SETUP (data-checkable items)
# ======================================================================================================================
f = "setup.tex"
idle2 = [r["idle_w"] for d in (P98B, P9LC) for r in d["runs"]]
C("second host idle 97 W", f + ":20", "97", mean(idle2), "mean of phase-9 idle_w (range %.2f-%.2f W)" % (min(idle2), max(idle2)))
vers = {r.get("version") for d in (A, REAL, LC1, MC1) for r in d["runs"] if r.get("version")}
B("engine versions vLLM 0.33 / SGLang 0.5.23", f + ":20", "0.33", ", ".join(sorted(vers)), vers == {"0.33.0", "0.5.23"}, "run 'version' fields")
uv = {c for d in (A, REAL, LC1) for r in d["runs"] for i, c in enumerate(r["command"]) if i and r["command"][i - 1] in ("--gpu-memory-utilization", "--mem-fraction-static")}
B("memory fractions 90% / 88%", f + ":21", "90", ", ".join(sorted(uv)), uv == {"0.90", "0.88"}, "engine command flags")
C("KV bytes/token Qwen3-32B = 256 KiB", f + ":25", "256", KV32 / 1024, "2 x 64 layers x 8 KV heads x 128 x 2 bytes (model calc from spec)")
C("KV bytes/token 30B-A3B = 96 KiB", f + ":26", "96", 2 * 48 * 4 * 128 * 2 / 1024, "2 x 48 x 4 x 128 x 2 bytes (model calc)")
C("KV bytes/token 8B = 144 KiB", f + ":26", "144", 2 * 36 * 8 * 128 * 2 / 1024, "2 x 36 x 8 x 128 x 2 bytes (model calc; 36 layers, 8 KV heads: spec)")
C("C Qwen3-32B vLLM", f + ":27", "244,928", CAP["vllm32"], "vLLM startup log")
C("C Qwen3-32B SGLang", f + ":28", "251,423", CAP["sgl32"], "SGLang startup log (max_total_num_tokens)")
C("C 30B-A3B vLLM", f + ":28", "704,032", CAP["moe"], "vLLM startup log")
C("C 8B vLLM", f + ":28", "777,872", CAP["vllm8"], "vLLM startup log")
C("C 8B SGLang", f + ":28", "780,230", CAP["sgl8"], "SGLang startup log")
t0p = [r["prompt_tokens"] for N_ in (16, 48, 96) for P in (5.0, 30.0) for s in sel(A, "vllm-gpu", N_, P) for r in rows(s) if r["turn"] == 0]
t0lc = [r["prompt_tokens"] for N_ in (4, 8, 16, 32) for s in sel(LC, "vllm-gpu-lc", N_) for r in rows(s) if r["turn"] == 0]
C("first prompt about 10.6k", f + ":39", "10.6", mean(t0p) / 1000, "mean turn-0 prompt, vLLM 11k GPU-cache steps, k tokens", about=True)
C("long-context first prompt about 26k", f + ":40", "26", mean(t0lc) / 1000, "mean turn-0 prompt, long-context GPU-cache steps", about=True)
outs = {r["output_tokens"] for s in sel(A, "vllm-gpu", 48) for r in rows(s)}
B("200-token answers", f + ":40", "200", str(sorted(outs)), outs == {200}, "output_tokens of every request (vLLM 48/5 s)")
grow = []
for N_ in (16, 48, 96):
    for s in sel(A, "vllm-gpu", N_):
        by = {}
        for r in rows(s):
            by.setdefault(r["session"], {})[r["turn"]] = r["prompt_tokens"]
        for d in by.values():
            grow += [d[k + 1] - d[k] for k in d if k + 1 in d]
C("prompt grows about 600 tokens per turn", f + ":41", "600", statistics.median(grow), "median consecutive-turn prompt growth, vLLM 11k steps", about=True)
durs = {d["settings"]["duration"] for d in (A, REAL, LC1, AD8)} | {d["settings"]["duration"] for d in (P98B, P9LC)}
B("150 s steps (600 s for 10-minute runs)", f + ":43", "150", str(sorted(durs)), durs == {150.0, 600.0}, "settings.duration")
B("drain 240 s, 15-min windows at 0 and 30 min", f + ":58", "240", f"drain {MC1['drain_s']}/{MCW2['drain_s']} s, window {MC1['window_s']} s, offsets {MC1['offset_s']}/{MCW2['offset_s']} s",
  MC1["drain_s"] == MCW2["drain_s"] == 240 and MC1["window_s"] == MCW2["window_s"] == 900 and MC1["offset_s"] == 0 and MCW2["offset_s"] == 1800, "mooncake JSON headers")
C("requests skipped (> 40,960 tokens), w1 1/4", f + ":56", "29", MC1["samples"]["0.25"]["skipped_too_long"], "samples['0.25'].skipped_too_long")
C("w1 1/4 sample size incl. skipped", f + ":56", "1,392", MC1["samples"]["0.25"]["requests"] + MC1["samples"]["0.25"]["skipped_too_long"], "requests + skipped")
mml = {r["command"][r["command"].index("--max-model-len") + 1] for r in MC1["runs"] if "--max-model-len" in r["command"]}
B("context limit 40,960 tokens", f + ":55", "40,960", ",".join(mml), mml == {"40960"}, "vLLM --max-model-len in the production runs")
B("BFCL 100 tasks at a time", f + ":61", "100", str(BFCL["settings"]["threads"]), BFCL["settings"]["threads"] == 100, "bfcl.json settings.threads")
B("BFCL 200 base + 200 long-context", f + ":60", "200", f"{BS['vllm-gpu-bfcl']['base'][1]} + {BS['vllm-gpu-bfcl']['long_context'][1]}",
  BS["vllm-gpu-bfcl"]["base"][1] == 200 and BS["vllm-gpu-bfcl"]["long_context"][1] == 200, "score file totals")

# ======================================================================================================================
# MODEL (Section 3): Eq. 1-6 recomputed from stated inputs
# ======================================================================================================================
f = "model.tex"
d_, h_, nq, nkv, hd, mlp, Ly = 5120, 5120, 64, 8, 128, 25600, 64
params = Ly * (d_ * nq * hd + 2 * d_ * nkv * hd + nq * hd * d_ + 3 * d_ * mlp)
C("non-embedding parameters 31.2 B", f + ":8", "31.2", params / 1e9, "64 x (q,k,v,o projections + 3 MLP matrices), Appendix shapes (spec)")
C("linear layers 62.4 GFLOP/token", f + ":8", "62.4", 2 * 31.2, "2 x 31.2e9")
C("attention 2.10 MFLOP per earlier token", f + ":9", "2.10", 2 * 2 * 64 * 128 * 64 / 1e6, "2 x 2 x 64 x 128 x 64")
# epsilon fit (Section 3): the ~10k recompute sends of Table 1 (both variants, 3 reps)
fit = [x for v in ("sglang-gpu-control", "sglang-host160") for x in tr_phase(v, "recompute", 8000)]
fit_tps = mean(x["prompt_tokens"] / x["duration_s"] for x in fit)
fit_w = mean(x["energy_j"] / x["duration_s"] for x in fit)
C("fit: 7,850 prompt tokens/s near 10k", f + ":19", "7,850", fit_tps, "mean prompt_tokens/duration of the 6 recompute sends at ~10k", about=True)
C("fit: 588 W", f + ":19", "588", fit_w, "mean energy_j/duration of the same sends")
idle_all = [r["idle_w"] for d in (A, OFF, H160, CTRL, REAL) for r in d["runs"] if r.get("idle_w")]
C("idle draw 92 W", f + ":21", "92", mean(idle_all), f"mean idle_w over first-host agent runs (range {min(idle_all):.2f}-{max(idle_all):.2f})")
TOKS, PW, IDLE_W, PEAK, EPS = 7850, 588.0, 92.0, 836e12, 0.87e-12  # printed inputs (peak: cited spec)
pf = TOKS * F(Lact[8000])
C("0.57 PFLOP/s", f + ":19", "0.57", pf / 1e15, "7,850 tok/s x F(L), L = 10,066 (mean length of the fitted sends); stated inputs")
C("about 68% of 836 TFLOP/s", f + ":19", "68", 100 * fit_tps * F(Lact[8000]) / PEAK, "unrounded fit data (0.57/0.836 as printed: %.1f%%; 7,850 x F(L): %.2f%%)" % (57 / 0.836, 100 * pf / PEAK), about=True)
C("epsilon 1.03 pJ/FLOP gross", f + ":20", "1.03", PW / pf * 1e12, "588 W / 0.57 PFLOP/s")
C("epsilon 0.87 pJ/FLOP above idle", f + ":20", "0.87", (PW - IDLE_W) / pf * 1e12, "(588 - 92) W / 0.57 PFLOP/s (unrounded data: %.3f)" % ((fit_w - IDLE_W) / (fit_tps * F(Lact[8000])) * 1e12))
for L, pr, ln in ((5000, "59", 21), (10000, "63", 21), (20000, "73", 21)):
    C(f"Eq.1 e_r at {L//1000}k = {pr} mJ", f + f":{ln}", pr, EPS * F(L) * 1e3, "0.87 pJ x F(L) for the stated 5k/10k/20k prompts")
C("measured recompute ~5k: 56-58", f + ":22", "56–58", (min(rec6[4000]), max(rec6[4000])), "Table 1, recompute sends of both variants, 3 reps")
C("measured recompute ~10k: 62-64", f + ":22", "62–64", (min(rec6[8000]), max(rec6[8000])), "Table 1")
C("measured recompute ~20k: 73", f + ":22", "73", (min(rec6[16000]) + max(rec6[16000])) / 2, "Table 1 (all 6 values round to 73: %.2f-%.2f)" % (min(rec6[16000]), max(rec6[16000])))
C("b_KV = 262,144 bytes", f + ":24", "262,144", KV32, "2 x 64 x 8 x 128 x 2")
C("copy time 5.2 us", f + ":25", "5.2", KV32 / 50e9 * 1e6, "262,144 B / 50 GB/s")
C("e_h = 0.5-1.3 mJ", f + ":28", "0.5–1.3", (100 * KV32 / 50e9 * 1e3, 250 * KV32 / 50e9 * 1e3), "P x b_KV / B for P = 100..250 W")
eh_lo, eh_hi = 100 * KV32 / 50e9 * 1e3, 250 * KV32 / 50e9 * 1e3
er = [EPS * F(L) * 1e3 for L in (5000, 10000, 20000)]
C("e_r/e_h about 45-140", f + ":31", "45–140", (min(er) / eh_hi, max(er) / eh_lo), "min e_r / max e_h .. max e_r / min e_h", about=True)
C("measured ratio 110-220 (model section)", f + ":31", "110–220", (ratio_mean[4000], ratio_mean[16000]), "ratio of means", about=True)
C("random-order cliff factor 1/(1-1/e)", f + ":48", "1.58", 1 / (1 - 1 / math.e), "mean-field model")
C("random-order 1.58 (cliff paragraph)", f + ":57", "1.58", 1 / (1 - 1 / math.e), "1/(1-1/e)")
C("5 s: half at W/C 1.2 (model text)", f + ":55", "1.2", cross[5.0][0], "as intro")
C("5 s: vanish by 1.55 (model text)", f + ":56", "1.55", cross[5.0][2], "as intro")
C("30 s: half at 1.5 (model text)", f + ":56", "1.5", cross[30.0][0], "as intro")
C("30 s: vanish by 2.2 (model text)", f + ":57", "2.2", cross[30.0][2], "as intro")
C("kappa band 1.2-2.2", f + ":57", "1.2–2.2", (cross[5.0][0], cross[30.0][2]), "as intro")
C("fit cell hit above GPU cache h = 0.754", f + ":69", "0.754", M(H160, "sglang-offload160", "cache_hit", 48) - SGH[(48, 5.0)], "160 GB 48/5 s hit - SGLang GPU hit, means of rounds")
C("fit cell 449.7 J", f + ":69", "449.7", M(H160, "sglang-offload160", "j_per_turn", 48), "mean of rounds")
C("fit cell E_thrash 1,058 J", f + ":69", "1,058", SG32[(48, 5.0)], "SGLang GPU cache 48/5 s, mean of rounds")
C("f = 0.76", f + ":69", "0.76", F_FIT, "(1 - E/E_thrash) / h")
# other overloaded synthetic Qwen3-32B host-tier cells (48 and 96 sessions), excluding 160 GB 48/30 s
tier_cells = [(A, "sglang-offload", n, p, "sglang") for n in (48, 96) for p in (5.0, 30.0)] + \
             [(OFF, "vllm-offload", n, p, "vllm") for n in (48, 96) for p in (5.0, 30.0)] + \
             [(H160, "sglang-offload160", 96, p, "sglang") for p in (5.0, 30.0)] + \
             [(SIZING, v, n, 5.0, "sglang") for v in ("sglang-host120", "sglang-host240", "sglang-host80-wb", "sglang-host160-wb") for n in (48, 96)] + \
             [(R3, "sglang-host160-r3", n, 5.0, "sglang") for n in (48, 96)]
tier_err = [abs(100 * (p / m_ - 1)) for doc, v, n, p_, b in tier_cells for p, m_ in [pred_tier(doc, v, n, p_, b)]]
C("Eq.5 predicts other overloaded 32B tier cells within 1.4%", f + ":70", "1.4", max(tier_err),
  f"max |pred/meas-1| over {len(tier_cells)} cells: SGLang 80 GB, vLLM 70 GiB (48/96 x 5/30 s), 160 GB 96, 120/240/80WB/160WB 48/96, 160 GB r3", kind="le")
c16 = [(A, "sglang-offload", 16, 5.0, "sglang"), (H160, "sglang-offload160", 16, 5.0, "sglang"), (OFF, "vllm-offload", 16, 5.0, "vllm"), (R3, "sglang-host160-r3", 16, 5.0, "sglang")]
err16 = [abs(100 * (p / m_ - 1)) for doc, v, n, p_, b in c16 for p, m_ in [pred_tier(doc, v, n, p_, b)]]
C("2.5-3.4% at 16 sessions", f + ":70", "2.5–3.4", (min(err16), max(err16)), "|pred/meas-1| at 16 sessions, 5 s pauses (80 GB, 160 GB, 160 GB r3, vLLM 70 GiB)")
p30, m30 = pred_tier(H160, "sglang-offload160", 48, 30.0)
C("exception: 160 GB 48/30 s -13%", f + ":71", "−13", 100 * (p30 / m30 - 1), "(pred - meas) / meas")
e8 = []
for N_ in (128, 256):
    Et, ht = M(P58B, "sglang-gpu-8b", "j_per_turn", N_), M(P58B, "sglang-gpu-8b", "cache_hit", N_)
    e8.append((Et * (1 - F_FIT * (M(P58B, "sglang-host160-8b", "cache_hit", N_) - ht)), M(P58B, "sglang-host160-8b", "j_per_turn", N_)))
C("underpredicts Qwen3-8B 160 GB cells by 4-5%", f + ":73", "4–5", tuple(sorted(-100 * (p / m_ - 1) for p, m_ in e8)), "E_thrash = SGLang GPU cache 8B (one round)")

# ======================================================================================================================
# SECTION 4: cost of a prompt token (Table 1)
# ======================================================================================================================
f = "results_cost.tex"
C("recompute 56-73 mJ", f + ":6", "56–73", (min(rec6[4000]), max(rec6[16000])), "Table 1 recompute range")
C("recompute rises 1.26x from ~5k to ~20k", f + ":7", "1.26", mean(rec9[16000]) / mean(rec9[4000]),
  "ratio of means of all nine full-recompute sends (both variants' recompute + control after-flush), as analysis.md (c)")
C("Eq.1 predicts 1.24x", f + ":7", "1.24", F(Lact[16000]) / F(Lact[4000]), "F(L) at measured mean prompt lengths")
C("restore 0.27-0.68 mJ", f + ":8", "0.27–0.68", (min(min(v) for v in rst.values()), max(max(v) for v in rst.values())), "host160 after-flush sends, all lengths, 3 reps")
C("predicted restore 0.5-1.3 mJ", f + ":9", "0.5–1.3", (eh_lo, eh_hi), "Eq. 2")
fr = [x["cached_frac"] * 100 for L in LENS for x in tr_phase("sglang-host160", "after_flush", L)]
C("99.98-100% of tokens from tier", f + ":9", "99.98–100", (min(fr), max(fr)), "cached_frac of restore sends")
C("GPU hit 0-0.3 mJ", f + ":9", "0–0.3", (min(hit_tr), max(hit_tr)), "gpu_hit sends, both variants, all lengths", unit=0.1)
C("recompute/restore ~110x at ~5k", f + ":10", "110", ratio_mean[4000], "ratio of means", about=True)
C("~160x at ~10k", f + ":10", "160", ratio_mean[8000], "ratio of means", about=True)
C("~220x at ~20k", f + ":10", "220", ratio_mean[16000], "ratio of means", about=True)
C("gap widens 2x", f + ":11", "2", ratio_mean[16000] / ratio_mean[4000], "ratio at ~20k / ratio at ~5k")
C("1.24x of e_r alone", f + ":11", "1.24", F(Lact[16000]) / F(Lact[4000]), "Eq. 1")
C("median TTFT restored, 20k", f + ":12", "1.36", statistics.median(x["ttft_p50"] for x in tr_phase("sglang-host160", "after_flush", 16000)), "median of per-rep TTFT medians")
C("median TTFT recomputed, 20k", f + ":12", "15.5", statistics.median(x["ttft_p50"] for v in ("sglang-gpu-control", "sglang-host160") for x in tr_phase(v, "recompute", 16000)), "median of per-rep TTFT medians")
ctl = [x["cached_frac"] * 100 for L in LENS for x in tr_phase("sglang-gpu-control", "after_flush", L)]
C("control: 0.1-0.4% of tokens cached", f + ":14", "0.1–0.4", (min(ctl), max(ctl)), "control after-flush cached_frac")
ctl_e = [abs(chg(a["above_idle_j_per_prompt_token"], b["above_idle_j_per_prompt_token"])) for L in LENS
         for a, b in zip(tr_phase("sglang-gpu-control", "after_flush", L), tr_phase("sglang-gpu-control", "recompute", L))]
C("control: energy within 4% of recompute", f + ":14", "4", max(ctl_e), "max |after_flush/recompute - 1| per rep and length, control variant", kind="le")
f2 = "results_cost.tex"
hd = [x["duration_s"] for L in LENS for v in ("sglang-gpu-control", "sglang-host160") for x in tr_phase(v, "gpu_hit", L)]
rd_ = [x["duration_s"] for L in LENS for x in tr_phase("sglang-host160", "after_flush", L)]
C("hit phases last 0.1-0.2 s", f2 + ":26", "0.1–0.2", (min(hd), max(hd)), "gpu_hit duration_s")
C("restore phases last 0.3-1.4 s", f2 + ":27", "0.3–1.4", (min(rd_), max(rd_)), "restore duration_s")
nneg = sum(1 for L in LENS for v in ("sglang-gpu-control", "sglang-host160") for x in tr_phase(v, "gpu_hit", L)
           if math.copysign(1, x["above_idle_j_per_prompt_token"]) < 0)
B("two hit readings slightly negative", f2 + ":27", "0.1", f"{nneg} negative", nneg == 2,
  "sign of above_idle_j_per_prompt_token of the gpu_hit sends")
C("per-repetition ratios 85-260x", f2 + ":28", "85–260", (min(ratio_rep), max(ratio_rep)), "mean recompute (n=9) / each rep's restore, 2 significant figures", about=True)
for L, pr, ln in ((4000, "56–58", 34), (8000, "62–64", 34), (16000, "73", 34)):
    if "–" in pr:
        C(f"Table 1 recompute {({4000: '~5k', 8000: '~10k', 16000: '~20k'})[L]}", f2 + f":{ln}", pr, (min(rec6[L]), max(rec6[L])), "recompute sends of both variants, per-rep range")
    else:
        C("Table 1 recompute ~20k (min)", f2 + f":{ln}", pr, min(rec6[L]), "all six sends round to 73")
        C("Table 1 recompute ~20k (max)", f2 + f":{ln}", pr, max(rec6[L]), "all six sends round to 73")
for L, pr in ((4000, "0.42–0.68"), (8000, "0.27–0.46"), (16000, "0.28–0.37")):
    C(f"Table 1 restore {({4000: '~5k', 8000: '~10k', 16000: '~20k'})[L]}", f2 + ":35", pr, (min(rst[L]), max(rst[L])), "host160 after-flush, 3 reps")
C("Table 1 GPU hit 0-0.3", f2 + ":36", "0–0.3", (min(hit_tr), max(hit_tr)), "all gpu_hit sends", unit=0.1)
for L, pr in ((4000, "110"), (8000, "160"), (16000, "220")):
    C(f"Table 1 ratio {({4000: '~5k', 8000: '~10k', 16000: '~20k'})[L]}", f2 + ":37", pr, ratio_mean[L], "ratio of means", about=True)
fill = [r["filled_prompt_tokens"] for r in IDLE["runs"]]
C("fill: 261k prompt tokens sent", f2 + ":43", "261", mean(fill) / 1000, "idle-cache.json filled_prompt_tokens (k)")
C("capacity 245-251k", f2 + ":43", "245–251", (CAP["vllm32"] / 1000, CAP["sgl32"] / 1000), "engine logs (k tokens)")
ffc = {r["variant"]: r["first_fill_prompt_cached_frac"] for r in IDLE["runs"]}
B("first fill prompt cached on SGLang, evicted on vLLM", f2 + ":44", "245", f"SGLang {ffc['sglang-idle']}, vLLM {ffc['vllm-idle']}",
  ffc["sglang-idle"] > 0.99 and ffc["vllm-idle"] == 0.0, "first_fill_prompt_cached_frac")
C("idle full vs empty, vLLM", f2 + ":45", "−0.08 to −0.15", tuple(idle_d["vllm-idle"]), "two readings (full, full-again)")
C("idle full vs empty, SGLang", f2 + ":45", "−0.13 to −0.16", tuple(idle_d["sglang-idle"]), "two readings")
C("rebuild 56-73 mJ (holding paragraph)", f2 + ":46", "56–73", (min(rec6[4000]), max(rec6[16000])), "Table 1")
C("moving costs well under 1 mJ", f2 + ":47", "1", max(max(v) for v in rst.values()), "max restore mJ", kind="le")


def e2(doc, eng, wl, u):
    e = [x for x in doc["engines"] if x["engine"] == eng][0]
    return [r for r in e["rows"] if r["workload"] == wl and r["users"] == u][0]


pen1 = [e2(E2L, e, "long_miss", 1)["joules_per_token"] / e2(E2L, e, "long_hit", 1)["joules_per_token"] for e in ("vllm", "sglang")]
pen64 = [e2(E2L, e, "long_miss", 64)["joules_per_token"] / e2(E2L, e, "long_hit", 64)["joules_per_token"] for e in ("vllm", "sglang")]
C("miss prompt 16k tokens", f2 + ":49", "16", e2(E2L, "vllm", "long_miss", 1)["prompt_tokens_median"] / 1000, "prompt_tokens_median (k)", about=True)
C("miss penalty at 1 user 1.46-1.48x", f2 + ":49", "1.46–1.48", (min(pen1), max(pen1)), "J per output token miss/hit, both engines, one round")
C("miss penalty at 64 users 10.1-10.6x", f2 + ":50", "10.1–10.6", (min(pen64), max(pen64)), "J per output token miss/hit, both engines")
B("64 miss contexts about four times the KV capacity", f2 + ":50", "64", f"{64 * e2(E2L, 'vllm', 'long_miss', 64)['prompt_tokens_median'] / CV:.2f}x",
      round(64 * e2(E2L, "vllm", "long_miss", 64)["prompt_tokens_median"] / CV) == 4, "64 x 16,027 tokens / C (rounds to 4)")

# ======================================================================================================================
# SECTION 5: the cliff (Tables 3, 4; Figure 2)
# ======================================================================================================================
f = "results_cliff.tex"
C("vLLM 16 sessions: 85% cached", f + ":5", "85", 100 * VGH[(16, 5.0)], "mean of rounds")
j16 = V(A, "vllm-gpu", "j_per_turn", 16) + V(R3, "vllm-gpu-r3", "j_per_turn", 16)
C("vLLM 16 sessions: 510-512 J", f + ":5", "510–512", (min(j16), max(j16)), "per-round range, rounds 1-3")
C("SGLang 16 sessions: 511-517 J", f + ":5", "511–517", (min(V(A, "sglang-gpu", "j_per_turn", 16)), max(V(A, "sglang-gpu", "j_per_turn", 16))), "per-round range")
hov = [x for N_ in (48, 96) for x in V(A, "vllm-gpu", "cache_hit", N_) + V(R3, "vllm-gpu-r3", "cache_hit", N_)]
C("48/96 sessions: hit 0.04% or less", f + ":6", "0.04", 100 * max(hov), "max step hit, vLLM 5 s, rounds 1-3", kind="le")
jov = [x for N_ in (48, 96) for x in V(A, "vllm-gpu", "j_per_turn", N_) + V(R3, "vllm-gpu-r3", "j_per_turn", N_)]
C("48/96 sessions: 1,024-1,052 J", f + ":6", "1,024–1,052", (min(jov), max(jov)), "per-round range, vLLM 5 s, rounds 1-3")
hos = [x for N_ in (48, 96) for x in V(A, "sglang-gpu", "cache_hit", N_)]
C("48/96 sessions: SGLang hit 0.12% or less", f + ":7", "0.12", 100 * max(hos), "max step hit, SGLang 5 s, rounds 1-2", kind="le")
jos = [x for N_ in (48, 96) for x in V(A, "sglang-gpu", "j_per_turn", N_)]
C("48/96 sessions: SGLang 1,035-1,058 J", f + ":7", "1,035–1,058", (min(jos), max(jos)), "per-round range, SGLang 5 s, rounds 1-2")
C("vLLM: within 0.5-1.6% of no cache", f + ":7", "0.5–1.6", (min(thr_vllm), max(thr_vllm)), "per-round |GPU/no cache - 1|, 4 cells x 2 rounds")
C("SGLang: within 0.01-0.15%", f + ":7", "0.01–0.15", (min(thr_sgl), max(thr_sgl)), "per-round, 4 cells x 2 rounds")
B("SGLang: identical turn counts", f + ":7", "0.15", "identical in all 8 cell-rounds" if sgl_same_turns else "differ", sgl_same_turns, "turns, GPU cache vs no cache")
C("vLLM gap at most 1.6% across rounds and seeds", f + ":8", "1.6", max(thr_vllm + thr_r3 + thr_seeds), "max |dJ| over rounds 1-3 and seeds s1/s2", kind="le")
r3dev = [abs(chg(M(R3, v3, "j_per_turn", N_), M(d, v, "j_per_turn", N_))) for N_ in (16, 48, 96)
         for v3, d, v in (("vllm-gpu-r3", A, "vllm-gpu"), ("vllm-nocache-r3", A, "vllm-nocache"), ("sglang-host160-r3", H160, "sglang-offload160"))]
C("third round repeated within 1.5%", f + ":9", "1.5", max(r3dev), "max |r3 / mean(r1,r2) - 1| J/turn, headline cells (GPU, no cache, 160 GB; 16/48/96)", kind="le")
ret_ok = all(s["cached_frac"] >= 0.998 for r in RET["runs"] for s in r["sends"] if s["send"] == "resend")
ret_max = max(s["delay_s"] for r in RET["runs"] for s in r["sends"])
B("resends after up to 25 min fully cached", f + ":10", "25", f"max delay {ret_max/60:.0f} min, min cached_frac {min(s['cached_frac'] for r in RET['runs'] for s in r['sends'] if s['send']=='resend')}",
  ret_ok and ret_max == 1500, "retention.json, both engines")
# Table 3
T3 = [("32B 48", R3, "vllm-gpu-r3", R3, "vllm-nocache-r3", 48, "0.03", "1,052", "1,069", "−1.5", 24),
      ("32B 96", R3, "vllm-gpu-r3", R3, "vllm-nocache-r3", 96, "0.03", "1,027", "1,040", "−1.2", 25),
      ("30B-A3B 96", MOE, "vllm-gpu-moe", MOE, "vllm-nocache-moe", 96, "0.02", "217.7", "218.5", "−0.3", 26),
      ("30B-A3B 192", MOE, "vllm-gpu-moe", MOE, "vllm-nocache-moe", 192, "0.01", "205.8", "206.9", "−0.5", 27),
      ("8B 128", P58B, "vllm-gpu-8b", P58B, "vllm-nocache-8b", 128, "0.03", "269.6", "270.4", "−0.3", 28),
      ("8B 256", P58B, "vllm-gpu-8b", P58B, "vllm-nocache-8b", 256, "0.01", "261.3", "261.8", "−0.2", 29),
      ("32B 27k 16", LC, "vllm-gpu-lc", LC, "vllm-nocache-lc", 16, "0.01", "2,964", "2,963", "+0.03", 30),
      ("32B 27k 32", LC, "vllm-gpu-lc", LC, "vllm-nocache-lc", 32, "0.01", "2,901", "2,902", "−0.04", 31)]
for lab, dg, g, dn, n, N_, ph, pg, pn, pd, ln in T3:
    rule = "round 3" if dg is R3 else ("mean of rounds (no cache: round 2)" if dg is LC else "mean of rounds")
    jg, jn = M(dg, g, "j_per_turn", N_), M(dn, n, "j_per_turn", N_)
    C(f"Table 3 {lab}: hit", f + f":{ln}", ph, 100 * M(dg, g, "cache_hit", N_), rule)
    C(f"Table 3 {lab}: cache J/turn", f + f":{ln}", pg, jg, rule)
    C(f"Table 3 {lab}: no-cache J/turn", f + f":{ln}", pn, jn, rule)
    C(f"Table 3 {lab}: delta", f + f":{ln}", pd, chg(jg, jn), "percent change of the printed means")
for N_, pr in ((16, "85"), (24, "20")):
    C(f"5 s hit at {N_} sessions", f + ":36", pr, hit_cliff(N_, 5.0), "mean of rounds")
h5 = [hit_cliff(N_, 5.0) for N_ in (32, 40, 48, 96)]
C("5 s hit at >= 32 sessions 0.01-0.04%", f + ":36", "0.01–0.04", (min(h5), max(h5)), "32/40/48/96 sessions")
for N_, pr in ((16, "67"), (24, "55"), (32, "29"), (40, "8.8"), (48, "0.05")):
    C(f"30 s hit at {N_} sessions", f + ":37", pr, hit_cliff(N_, 30.0), "mean of rounds")
rd_diff = max(abs(100 * (a - b)) for N_ in (16, 24, 32, 40, 48, 96) for P in (5.0, 30.0)
              for a, b in [V(A if N_ in (16, 48, 96) else CLIFF, "vllm-gpu", "cache_hit", N_, P)[:2]])
C("both rounds agree within 0.3 points", f + ":38", "0.3", rd_diff, "max |hit r1 - hit r2| (pp), vLLM cliff cells", kind="le")
C("5 s half at 1.2 (cliff section)", f + ":38", "1.2", cross[5.0][0], "interpolated W/C")
C("5 s vanish by 1.55", f + ":38", "1.55", cross[5.0][2], "first W/C at ~0")
C("30 s half at 1.5", f + ":39", "1.5", cross[30.0][0], "interpolated W/C")
C("30 s vanish by 2.2", f + ":39", "2.2", cross[30.0][2], "first W/C at ~0")
C("random-order model kappa = 1.58", f + ":39", "1.58", 1 / (1 - 1 / math.e), "1/(1-1/e)")
C("still 8.8% at W/C", f + ":39", "1.86", WC_40_30, "W/C of the 40-session, 30 s step")
C("still 8.8% hit", f + ":39", "8.8", hit_cliff(40, 30.0), "hit at that point")
mm = []
for N_ in (16, 48, 96):
    for P in (5.0, 30.0):
        for s in sel(A, "vllm-gpu", N_, P):
            by = {}
            for r in rows(s):
                by.setdefault(r["session"], []).append(r["prompt_tokens"])
            mm.append(mean(max(v) / mean(v) for v in by.values()))
C("W uses largest prompt, up to about 20% above mean", f + ":42", "20", 100 * (max(mm) - 1), "max over cliff steps of mean_session(max prompt / mean prompt) - 1", about=True)
C("30B-A3B KV per token 96 KiB", f + ":45", "96", 2 * 48 * 4 * 128 * 2 / 1024, "model calc")
C("30B-A3B capacity 704k", f + ":45", "704", CAP["moe"] / 1000, "vLLM log (k)")
C("30B-A3B capacity 2.9x dense 32B", f + ":46", "2.9", CAP["moe"] / CV, "ratio of logged capacities")
C("30B-A3B 91% hits at 32", f + ":46", "91", 100 * M(MOE, "vllm-gpu-moe", "cache_hit", 32), "mean of rounds")
C("30B-A3B 147 J at 32", f + ":46", "147", M(MOE, "vllm-gpu-moe", "j_per_turn", 32), "mean of rounds")
C("30B-A3B 279 J without cache", f + ":46", "279", M(MOE, "vllm-nocache-moe", "j_per_turn", 32), "mean of rounds")
C("30B-A3B -47%", f + ":46", "−47", chg(M(MOE, "vllm-gpu-moe", "j_per_turn", 32), M(MOE, "vllm-nocache-moe", "j_per_turn", 32)), "percent change")
B("30B-A3B collapsed by 96", f + ":47", "96", f"hit {100*M(MOE,'vllm-gpu-moe','cache_hit',96):.2f}%", M(MOE, "vllm-gpu-moe", "cache_hit", 96) < 0.01, "hit < 1%")
s_moe = M(MOE, "vllm-gpu-moe", "context_tokens_p50", 96)
C("30B-A3B median session 11.8k", f + ":48", "11.8", s_moe / 1000, "context_tokens_p50 at 96 sessions (k)")
C("30B-A3B C/s about 60", f + ":48", "60", CAP["moe"] / s_moe, "704,032 / median session", about=True)
C("30B-A3B 1.2 C/s about 72", f + ":47", "72", 1.2 * CAP["moe"] / s_moe, "1.2 x C/s")
C("30B-A3B 1.55 C/s about 93", f + ":47", "93", 1.55 * CAP["moe"] / s_moe, "1.55 x C/s")
C("Qwen3-8B capacity 778k", f + ":48", "778", CAP["vllm8"] / 1000, "vLLM log (k)")
C("Qwen3-8B 60.5% at 64", f + ":48", "60.5", 100 * M(P58B, "vllm-gpu-8b", "cache_hit", 64), "mean of rounds")
B("Qwen3-8B collapsed by 128", f + ":49", "128", f"hit {100*M(P58B,'vllm-gpu-8b','cache_hit',128):.2f}%", M(P58B, "vllm-gpu-8b", "cache_hit", 128) < 0.01, "hit < 1%")
d_moe8 = [abs(chg(M(d, g, "j_per_turn", N_), M(d, n, "j_per_turn", N_))) for d, g, n, Ns in ((MOE, "vllm-gpu-moe", "vllm-nocache-moe", (96, 192)), (P58B, "vllm-gpu-8b", "vllm-nocache-8b", (128, 256))) for N_ in Ns]
C("30B-A3B and 8B past cliff within 0.5% of no cache", f + ":49", "0.5", max(d_moe8), "max |delta| of the Table 3 means", kind="le")
C("long-context start about 26k", f + ":52", "26", mean(t0lc) / 1000, "mean turn-0 prompt (k)", about=True)
lcm = [s["prompt_tokens"] / s["turns"] for N_ in (4, 8, 16, 32) for s in sel(LC, "vllm-gpu-lc", N_)]
C("long-context average 26-29k per request", f + ":52", "26–29", (min(lcm) / 1000, max(lcm) / 1000), "prompt_tokens/turns per step (k)")
C("long-context hit at 4", f + ":53", "90.5", 100 * M(LC, "vllm-gpu-lc", "cache_hit", 4), "mean of rounds")
C("long-context hit at 8", f + ":53", "88.3", 100 * M(LC, "vllm-gpu-lc", "cache_hit", 8), "mean of rounds")
C("pre-registered band low end 1.2 C/s ~ 12", f + ":54", "12", 1.2 * CV / 24000, "C = 244,928, s = 24k (plan estimate)")
C("pre-registered band high end 2 C/s ~ 20", f + ":54", "20", 2 * CV / 24000, "2 C / 24k")
C("measured 26-29k gives 10-19", f + ":55", "10–19", (1.2 * CV / (max(lcm)), 2 * CV / (min(lcm))), "1.2 C / s_max .. 2 C / s_min with s = per-step mean prompt")
sv = [-chg(M(LC, "vllm-gpu-lc", "j_per_turn", N_), M(LC, "vllm-nocache-lc", "j_per_turn", N_)) for N_ in (4, 8)]
mt = [chg(M(LC, "vllm-gpu-lc", "turns", N_), M(LC, "vllm-nocache-lc", "turns", N_)) for N_ in (4, 8)]
C("before the wall cache cut energy 57-64%", f + ":55", "57–64", (min(sv), max(sv)), "4 and 8 sessions, GPU mean of rounds vs no cache (round 2)")
C("before the wall 81-150% more turns", f + ":56", "81–150", (min(mt), max(mt)), "turns")
pw = [abs(chg(M(LC, "vllm-gpu-lc", "j_per_turn", N_), M(LC, "vllm-nocache-lc", "j_per_turn", N_))) for N_ in (16, 32)]
C("past the wall differ by at most 0.04%", f + ":56", "0.04", max(pw), "16 and 32 sessions", kind="le")
C("sessions about 2.5x longer", f + ":57", "2.5", mean(t0lc) / mean(t0p), "mean first prompt, long-context / 11k runs")
lc_rd = [abs(chg(a, b)) for N_ in (4, 8, 16, 32) for a, b in [V(LC, "vllm-gpu-lc", "j_per_turn", N_)]]
C("both rounds agree within 0.1%", f + ":58", "0.1", max(lc_rd), "GPU-cache J/turn r1 vs r2", kind="le")
# Table 4
for N_, ph, pj, pn, pt, pp, ln in ((4, "90.5", "1,495", "3,447", "47", "0.27–0.28", 70), (8, "88.3", "1,095", "3,039", "75", "0.35–0.36", 71),
                                   (16, "0.01", "2,964", "2,963", "37", "40", 72), (32, "0.01", "2,901", "2,902", "54", "121", 73)):
    C(f"Table 4 N={N_}: hit", f + f":{ln}", ph, 100 * M(LC, "vllm-gpu-lc", "cache_hit", N_), "mean of rounds")
    C(f"Table 4 N={N_}: J/turn cache", f + f":{ln}", pj, M(LC, "vllm-gpu-lc", "j_per_turn", N_), "mean of rounds")
    C(f"Table 4 N={N_}: J/turn no cache", f + f":{ln}", pn, M(LC, "vllm-nocache-lc", "j_per_turn", N_), "round 2")
    C(f"Table 4 N={N_}: turns", f + f":{ln}", pt, M(LC, "vllm-gpu-lc", "turns", N_), "mean of rounds")
    pv = V(LC, "vllm-gpu-lc", "resume_ttft_p90", N_)
    C(f"Table 4 N={N_}: p90 resume", f + f":{ln}", pp, (min(pv), max(pv)) if "–" in pp else mean(pv), "range over rounds" if "–" in pp else "both rounds")
# Figure 2 (make_figures.py series): x = W/C (mean over rounds), y = GPU-cache / no-cache J per turn
wall_series = [(R3, "vllm-gpu-r3", "vllm-nocache-r3", (16, 48, 96), CV), (LC, "vllm-gpu-lc", "vllm-nocache-lc", (4, 8, 16, 32), CV),
               (P58B, "vllm-gpu-8b", "vllm-nocache-8b", (64, 128, 256), CAP["vllm8"]), (MOE, "vllm-gpu-moe", "vllm-nocache-moe", (32, 96, 192), CAP["moe"])]
wall_pts = [(mean(working_set(s) / cap for s in sel(d, g, N_)), M(d, g, "j_per_turn", N_) / M(d, n, "j_per_turn", N_))
            for d, g, n, Ns, cap in wall_series for N_ in Ns]
below = [y for x, y in wall_pts if x <= 1.25]
past = [y for x, y in wall_pts if x >= 1.7]
B("Fig. 2: no points between W/C 1.25 and 1.7", f + ":83", "1.75", "W/C of points: " + ", ".join(f"{x:.2f}" for x, _ in sorted(wall_pts)),
  len(below) + len(past) == len(wall_pts), "every plotted point is either <= ~1.2 or >= ~1.75")
C("Fig. 2: below-cliff points reach W/C about 1.2", f + ":83", "1.2", max(x for x, y in wall_pts if x <= 1.25), "largest W/C of the below-cliff points")
C("Fig. 2: post-cliff points from W/C 1.75", f + ":83", "1.75", min(x for x, y in wall_pts if x >= 1.7), "smallest W/C of the post-cliff points")
C("Fig. 2: below the cliff the cache saves 38-64%", f + ":83", "38–64", (100 * (1 - max(below)), 100 * (1 - min(below))), "points with W/C <= 1.2")
C("Fig. 2: past it ratio is 1 within 2%", f + ":83", "2", 100 * max(abs(x - 1) for x in past), "max |ratio - 1| of points with W/C >= 1.75", kind="le")
C("Fig. 2: shaded band kappa 1.2-2.2", f + ":84", "1.2–2.2", (cross[5.0][0], cross[30.0][2]), "as intro")
ev = [abs(chg(M(EVICT, v, "j_per_turn", N_, P), SG32[(N_, P)])) for v in ("sglang-lfu", "sglang-slru") for N_ in (48, 96) for P in (5.0, 30.0)]
C("LFU/SLRU within 0.2% of LRU", f + ":87", "0.2", max(ev), "48/96 sessions, both pauses, means of rounds", kind="le")
evr = [chg(a, b) for v in ("sglang-lfu-real", "sglang-slru-real") for a, b in zip(V(REVICT, v, "j_per_turn", 48), V(REAL, "sglang-gpu-real", "j_per_turn", 48))]
C("real trajectories: LFU/SLRU 20-28% more", f + ":88", "20–28", (min(evr), max(evr)), "48 sessions, round-matched vs SGLang LRU")
for N_, pr, ln in ((16, "88", 92), (48, "85", 92), (96, "68", 92)):
    C(f"real trajectories GPU hit at {N_}", f + f":{ln}", pr, 100 * M(REAL, "vllm-gpu-real", "cache_hit", N_), "vLLM, mean of rounds")
hh = V(REALHI, "vllm-gpu-real-hi", "cache_hit", 192) + V(REALHI, "vllm-gpu-real-hi", "cache_hit", 288)
C("real trajectories 72-73% at 192/288", f + ":92", "72–73", (100 * min(hh), 100 * max(hh)), "per-round range")

# ======================================================================================================================
# SECTION 6: host tiers (Table 2)
# ======================================================================================================================
f = "results_tiers.tex"
C("Table 2 caption: GPU alone 48 sessions hit", f + ":12", "0.12", 100 * SGH[(48, 5.0)], "SGLang GPU cache, mean of rounds")
C("Table 2 caption: GPU alone 1,058 J", f + ":12", "1,058", SG32[(48, 5.0)], "mean of rounds")
C("Table 2 caption: GPU alone at 96 hit 0.09%", f + ":12", "0.09", 100 * SGH[(96, 5.0)], "SGLang GPU cache, mean of rounds")
C("Table 2 caption: GPU alone at 96 1,035 J", f + ":12", "1,035", SG32[(96, 5.0)], "mean of rounds")
T2 = [("80 GB WT 48", A, "sglang-offload", 48, "11.4–12.6", "953–962", "−10 to −9", 18, "range"),
      ("80 GB WB 48", SIZING, "sglang-host80-wb", 48, "36.5", "768", "−27", 19, "mean"),
      ("120 GB WT 48", SIZING, "sglang-host120", 48, "49.9–50.3", "652–655", "−38", 20, "mean"),
      ("160 GB WT 48", H160, "sglang-offload160", 48, "75.5", "450", "−57", 21, "mean"),
      ("240 GB WT 48", SIZING, "sglang-host240", 48, "75.5", "448–449", "−58", 22, "mean"),
      ("160 GB WT 96", H160, "sglang-offload160", 96, "20.2", "873", "−16", 23, "mean"),
      ("240 GB WT 96", SIZING, "sglang-host240", 96, "37.2", "732", "−29", 24, "mean")]
for lab, d, v, N_, ph, pj, pv, ln, _ in T2:
    hv, jv = [100 * x for x in V(d, v, "cache_hit", N_)], V(d, v, "j_per_turn", N_)
    C(f"Table 2 {lab}: hit", f + f":{ln}", ph, (min(hv), max(hv)) if "–" in ph else mean(hv), "per-round range" if "–" in ph else "mean of rounds")
    C(f"Table 2 {lab}: J/turn", f + f":{ln}", pj, (min(jv), max(jv)) if "–" in pj else mean(jv), "per-round range" if "–" in pj else "mean of rounds")
    vs = [chg(x, SG32[(N_, 5.0)]) for x in jv]
    C(f"Table 2 {lab}: vs GPU", f + f":{ln}", pv, (min(vs), max(vs)) if " to " in pv else chg(mean(jv), SG32[(N_, 5.0)]),
      "per-round J vs GPU mean" if " to " in pv else "mean vs GPU mean")
C("working set at 48 sessions about 548k", f + ":29", "548", W48 / 1000, "sum of largest prompt per session, vLLM 48/5 s (k)", about=True)
C("80 GB holds about 305k", f + ":31", "305", CAP["host80"] / 1000, "SGLang log 'KV host pool' (k)", about=True)
C("80 GB W/C about 1.8", f + ":31", "1.8", W48 / CAP["host80"], "W / host tokens")
s80 = [-chg(x, SG32[(48, 5.0)]) for x in V(A, "sglang-offload", "j_per_turn", 48)]
C("80 GB saved 9-10%", f + ":31", "9–10", (min(s80), max(s80)), "per round vs GPU mean")
C("160 GB holds about 610k", f + ":32", "610", CAP["host160"] / 1000, "SGLang log (k)", about=True)
C("160 GB W/C about 0.9", f + ":32", "0.9", W48 / CAP["host160"], "W / host tokens")
C("160 GB saved 57%", f + ":32", "57", -chg(M(H160, "sglang-offload160", "j_per_turn", 48), SG32[(48, 5.0)]), "means of rounds")
C("160 GB 84% more turns", f + ":32", "84", chg(M(H160, "sglang-offload160", "turns", 48), M(A, "sglang-gpu", "turns", 48)), "turns, means of rounds")
C("160 GB median TTFT 9 s", f + ":32", "9", M(H160, "sglang-offload160", "resume_ttft_p50", 48), "resume_ttft_p50, mean of rounds")
C("GPU cache median TTFT 55 s", f + ":33", "55", M(A, "sglang-gpu", "resume_ttft_p50", 48), "resume_ttft_p50, SGLang GPU, mean of rounds")
B("240 GB added nothing at 48 sessions", f + ":33", "240", f"{chg(M(SIZING,'sglang-host240','j_per_turn',48), M(H160,'sglang-offload160','j_per_turn',48)):+.2f}% J/turn vs 160 GB",
      abs(chg(M(SIZING, "sglang-host240", "j_per_turn", 48), M(H160, "sglang-offload160", "j_per_turn", 48))) < 0.5, "|240 GB / 160 GB - 1| < 0.5%")
C("240 GB saved 29% at 96", f + ":33", "29", -chg(M(SIZING, "sglang-host240", "j_per_turn", 96), SG32[(96, 5.0)]), "means of rounds")
v70 = [chg(M(OFF, "vllm-offload", "j_per_turn", N_, P), G32[(N_, P)]) for N_ in (16, 48, 96) for P in (5.0, 30.0)]
C("vLLM 70 GiB changed energy +0.3% to -6%", f + ":34", "+0.3% to −6", (min(v70), max(v70)), "all 6 cells, means of rounds")
C("WB: hit 11.4-12.6% -> 36.5%", f + ":37", "11.4–12.6", (100 * min(V(A, "sglang-offload", "cache_hit", 48)), 100 * max(V(A, "sglang-offload", "cache_hit", 48))), "80 GB WT per round")
C("WB hit 36.5%", f + ":38", "36.5", 100 * M(SIZING, "sglang-host80-wb", "cache_hit", 48), "mean of rounds")
C("WB cut energy 27%", f + ":38", "27", -chg(M(SIZING, "sglang-host80-wb", "j_per_turn", 48), SG32[(48, 5.0)]), "vs GPU mean")
C("instead of 9-10%", f + ":38", "9–10", (min(s80), max(s80)), "80 GB WT")
w80 = [M(P5REAL, "sglang-host80wb-real", "j_per_turn", N_) for N_ in (192, 288)]
w160 = [M(REALHI, "sglang-host160-real-hi", "j_per_turn", N_) for N_ in (192, 288)]
C("80 GB WB within 1.3-4.5% of 160 GB WT (real)", f + ":39", "1.3–4.5", tuple(sorted(chg(a, b) for a, b in zip(w80, w160))), "J/turn, 192 and 288 sessions, means")
C("68.8 J (80 GB WB, 192)", f + ":40", "68.8", w80[0], "mean of rounds")
C("67.9 J (160 GB, 192)", f + ":40", "67.9", w160[0], "mean of rounds")
C("62.1 J (80 GB WB, 288)", f + ":40", "62.1", w80[1], "mean of rounds")
C("59.4 J (160 GB, 288)", f + ":40", "59.4", w160[1], "mean of rounds")
tdiff = [abs(chg(M(P5REAL, "sglang-host80wb-real", "turns", N_), M(REALHI, "sglang-host160-real-hi", "turns", N_))) for N_ in (192, 288)]
C("turn counts within 4.5%", f + ":40", "4.5", max(tdiff), "|turns 80 WB / 160 WT - 1|", kind="le")
e96 = M(REAL, "sglang-host160-real", "j_per_turn", 96)
g96 = V(REAL, "sglang-gpu-real", "j_per_turn", 96)
C("real 96: 160 GB lowered energy 33%", f + ":43", "33", -chg(e96, mean(g96)), "vs SGLang GPU cache, means")
C("real 96: 33% more turns", f + ":44", "33", chg(M(REAL, "sglang-host160-real", "turns", 96), M(REAL, "sglang-gpu-real", "turns", 96)), "turns")
C("real 96: 91.5 J", f + ":44", "91.5", e96, "mean of rounds")
C("real 96: GPU 136.6-138.2 J", f + ":44", "136.6–138.2", (min(g96), max(g96)), "per-round range")
for N_, ps, pt, ln in ((192, "27", "22", 44), (288, "31", "27", 45)):
    C(f"real {N_}: 160 GB saved vs vLLM GPU", f + f":{ln}", ps, -chg(M(REALHI, "sglang-host160-real-hi", "j_per_turn", N_), M(REALHI, "vllm-gpu-real-hi", "j_per_turn", N_)), "means")
    C(f"real {N_}: more turns", f + f":{ln + 1}", pt, chg(M(REALHI, "sglang-host160-real-hi", "turns", N_), M(REALHI, "vllm-gpu-real-hi", "turns", N_)), "means")
C("real 48: saved 3%", f + ":47", "3", -chg(M(REAL, "sglang-host160-real", "j_per_turn", 48), M(REAL, "sglang-gpu-real", "j_per_turn", 48)), "vs SGLang GPU cache")
C("f = 0.76 (tiers)", f + ":50", "0.76", F_FIT, "fit")
p80, m80 = pred_tier(A, "sglang-offload", 48, 5.0)
C("80 GB predicted 962 J", f + ":51", "962", p80, "Eq. 5, f fitted")
C("80 GB measured 958 J", f + ":51", "958", m80, "mean of rounds")
for P, pp, pm, ln in ((5.0, "1,035", "1,035", 51), (30.0, "973", "973", 52)):
    p_, m_ = pred_tier(OFF, "vllm-offload", 48, P, "vllm")
    C(f"vLLM 70 GiB 48/{P:g} s predicted", f + f":{ln}", pp, p_, "Eq. 5 with vLLM GPU cache as E_thrash")
    C(f"vLLM 70 GiB 48/{P:g} s measured", f + f":{ln}", pm, m_, "mean of rounds")
C("other overloaded cells within 1.4% (tiers)", f + ":52", "1.4", max(tier_err), "as Section 3", kind="le")
C("2.5-3.4% at 16 sessions (tiers)", f + ":52", "2.5–3.4", (min(err16), max(err16)), "as Section 3")
C("160 GB 48/30 s predicted 534 J", f + ":54", "534", p30, "Eq. 5")
C("160 GB 48/30 s measured 614 J", f + ":54", "614", m30, "mean of rounds")
C("160 GB 48/30 s drew 563 W", f + ":54", "563", M(H160, "sglang-offload160", "avg_w", 48, 30.0), "avg_w, mean of rounds")
C("160 GB 48/5 s drew 578 W", f + ":54", "578", M(H160, "sglang-offload160", "avg_w", 48, 5.0), "avg_w, mean of rounds")
C("160 GB 48/30 s error -13%", f + ":54", "−13", 100 * (p30 / m30 - 1), "(pred-meas)/meas")
for (p_, m_), pp, pm in zip(e8, ("207", "254"), ("215", "266")):
    C(f"Qwen3-8B 160 GB predicted {pp}", f + ":55", pp, p_, "Eq. 5, E_thrash = SGLang GPU cache 8B (r1)")
    C(f"Qwen3-8B 160 GB measured {pm}", f + ":56", pm, m_, "mean of rounds")
C("Qwen3-8B error -4 to -5%", f + ":56", "−4 to −5", tuple(100 * (p / m_ - 1) for p, m_ in e8), "(pred-meas)/meas")
rt = []
for N_ in (96, 192, 288):
    dn, dt_ = (REAL, REAL) if N_ == 96 else (REALHI, REALHI)
    vn, vt = ("vllm-nocache-real", "sglang-host160-real") if N_ == 96 else ("vllm-nocache-real-hi", "sglang-host160-real-hi")
    Et = M(dn, vn, "j_per_turn", N_)
    rt.append(-100 * (Et * (1 - F_FIT * M(dt_, vt, "cache_hit", N_)) / M(dt_, vt, "j_per_turn", N_) - 1))
C("recorded trajectories: predicts 6-23% below", f + ":57", "6–23", (min(rt), max(rt)),
  "E_thrash = vLLM no cache, 160 GB cells at 96, 192, 288 sessions (cells under memory pressure)")
hc = HOST
C("host bound 160 GB at 48/96: 0.5-1.8%", f + ":62", "0.5–1.8", (min(hc[("sgl160", n, p)] for n in (48, 96) for p in (5.0, 30.0)), max(hc[("sgl160", n, p)] for n in (48, 96) for p in (5.0, 30.0))),
  "(max(extra CPU,0) x 10 W/core + bytes x 1 nJ) / GPU J saved; CPU from cpu-time*.csv (port of analyze.py sec. 4)")
C("host bound 80 GB at 48: 2.9-3.0%", f + ":62", "2.9–3.0", (min(hc[("sgl80", 48, p)] for p in (5.0, 30.0)), max(hc[("sgl80", 48, p)] for p in (5.0, 30.0))), "same bound")
set11 = [x for x in (hc[("sgl80", 16, 5.0)], hc[("sgl160", 16, 5.0)], hc[("sgl80", 96, 5.0)], hc[("sgl80", 96, 30.0)]) if x is not None]
C("host bound 11-14% (16 sessions, 80 GB at 96)", f + ":63", "11–14", (min(set11), max(set11)), "SGLang 80/160 GB at 16/5 s and 80 GB at 96 (16/30 s 160 GB: no saving)")
C("host bound 128% (80 GB, 16/30 s)", f + ":64", "128", hc[("sgl80", 16, 30.0)], "same bound")
v70h = [x for x in (hc[("vllm70", n, p)] for n in (16, 48, 96) for p in (5.0, 30.0)) if x is not None]
C("host bound vLLM 70 GiB 4.7-212%", f + ":64", "4.7–212", (min(v70h), max(v70h)), "cells with a GPU saving")

# ======================================================================================================================
# SECTION 7: production traffic (Table 5)
# ======================================================================================================================
f = "results_production.tex"
C("w1 1/4 requests", f + ":10", "1,363", W1["vllm-gpu-mc"]["requests"], "replayed requests")
C("w2 1/4 requests", f + ":10", "1,542", W2["vllm-gpu-mc"]["requests"], "replayed requests")
C("11.5-12.2k prompt tokens/s", f + ":11", "11.5–12.2", (MC1["samples"]["0.25"]["prompt_tok_per_s"] / 1000, MCW2["samples"]["0.25"]["prompt_tok_per_s"] / 1000), "sample headers")
C("GPU cache alone finished 85-86%", f + ":12", "85–86", (min(gpu_done), max(gpu_done)), "vLLM w1, SGLang w1, vLLM w2")
mt_gpu = [W1["vllm-gpu-mc"]["ttft_p50"], W1["sglang-gpu-mc"]["ttft_p50"], W2["vllm-gpu-mc"]["ttft_p50"]]
C("GPU cache median TTFT 89-200 s", f + ":13", "89–200", (min(mt_gpu), max(mt_gpu)), "ttft_p50")
nc_done = [done(W1["vllm-nocache-mc"]), done(W1R2["vllm-nocache-mc"]), done(W2["vllm-nocache-mc"])]
C("no cache finished 61-62%", f + ":13", "61–62", (min(nc_done), max(nc_done)), "w1 r1, r2, w2")
C("tier 15-17% less J/prompt token than GPU cache", f + ":14", "15–17", (min(tier_less_mj), max(tier_less_mj)), "as intro")
nc_less = [-chg(mjt(W1["sglang-host160-mc"]), mjt(W1["vllm-nocache-mc"])), -chg(mjt(W2["sglang-host160-mc"]), mjt(W2["vllm-nocache-mc"]))]
C("about 40% less than no cache", f + ":14", "40", mean(nc_less), "mean of w1, w2 (%.1f, %.1f)" % tuple(nc_less), about=True)
pw_ = [s["avg_w"] for d in (W1, W1R2, W2) for s in d.values()]
C("card averaged 585-590 W", f + ":15", "585–590", (min(pw_), max(pw_)), "avg_w of every 1/4 cell, both windows and rounds")
etot = []
for d in (W1, W2):
    es = [d[v]["energy_j"] for v in d if v in ("vllm-nocache-mc", "vllm-gpu-mc", "sglang-gpu-mc", "sglang-host160-mc")]
    etot.append(100 * (max(es) / min(es) - 1))
C("total energy differed by at most 3.3%", f + ":16", "3.3", max(etot), "(max/min - 1) of window energy over no cache, GPU (both engines), 160 GB", kind="le")
C("16-18% more prompt tokens finished", f + ":16", "16–18", (min(tier_more_tok), max(tier_more_tok)), "as abstract")
C("1/8: 160 GB 85.54 mJ", f + ":17", "85.54", mjt(W2E["sglang-host160-mc"]), "w2 1/8")
C("1/8: GPU 86.08 mJ", f + ":17", "86.08", mjt(W2E["vllm-gpu-mc"]), "w2 1/8")
C("1/8: -0.6%", f + ":18", "−0.6", chg(mjt(W2E["sglang-host160-mc"]), mjt(W2E["vllm-gpu-mc"])), "percent change")
C("vLLM vs SGLang GPU cache: 0.1 points hit", f + ":18", "0.1", 100 * abs(W1["vllm-gpu-mc"]["cache_hit"] - W1["sglang-gpu-mc"]["cache_hit"]), "w1 r1")
C("vLLM vs SGLang GPU cache: 1% energy", f + ":18", "1", abs(chg(mjt(W1["sglang-gpu-mc"]), mjt(W1["vllm-gpu-mc"]))), "J per prompt token, w1 r1")
C("160 GB repeated within 0.1%", f + ":19", "0.1", abs(chg(mjt(W1R2["sglang-host160-mc"]), mjt(W1["sglang-host160-mc"]))), "r2 vs r1", kind="le")
B("300 GB raised hit by three points", f + ":19", "300", f"{100*(W1['sglang-host300-mc']['cache_hit']-W1['sglang-host160-mc']['cache_hit']):.2f} pp",
      round(100 * (W1["sglang-host300-mc"]["cache_hit"] - W1["sglang-host160-mc"]["cache_hit"])) == 3, "rounds to 3")
C("300 GB lowered energy another 4%", f + ":20", "4", -chg(mjt(W1["sglang-host300-mc"]), mjt(W1["sglang-host160-mc"])), "vs 160 GB")
C("300 GB median TTFT from 129 s", f + ":20", "129", W1["sglang-host160-mc"]["ttft_p50"], "160 GB ttft_p50")
C("300 GB median TTFT to 107 s", f + ":20", "107", W1["sglang-host300-mc"]["ttft_p50"], "300 GB ttft_p50")
C("Table 5 caption: drain 240 s", f + ":25", "240", MC1["drain_s"], "header")
r2c = []
for v in ("vllm-gpu-mc", "vllm-nocache-mc", "sglang-host160-mc"):
    a, b = W1[v], W1R2[v]
    r2c.append(("e", abs(chg(mjt(b), mjt(a)))))
    if a["cache_hit"]:
        r2c.append(("h", abs(chg(b["cache_hit"], a["cache_hit"]))))
    r2c.append(("t", abs(chg(b["ttft_p50"], a["ttft_p50"]))))
C("Table 5: second round within 0.2% energy and hit", f + ":26", "0.2", max(x for k, x in r2c if k in "eh"), "|r2/r1-1| J/prompt token and hit, 3 cells", kind="le")
C("Table 5: second round within 0.3% TTFT", f + ":26", "0.3", max(x for k, x in r2c if k == "t"), "|r2/r1-1| median TTFT", kind="le")
T5 = [("0–15 no cache (vLLM)", W1["vllm-nocache-mc"], "61", "0", "104.4", "270", 32),
      ("0–15 GPU (vLLM)", W1["vllm-gpu-mc"], "86", "36.7", "74.8", "200", 33),
      ("0–15 GPU (SGLang)", W1["sglang-gpu-mc"], "86", "36.6", "75.6", "199", 34),
      ("0–15 160 GB", W1["sglang-host160-mc"], "100", "51.8", "62.8", "129", 35),
      ("0–15 300 GB", W1["sglang-host300-mc"], "100", "54.8", "60.3", "107", 36),
      ("30–45 no cache (vLLM)", W2["vllm-nocache-mc"], "62", "0", "101.6", "236", 37),
      ("30–45 GPU (vLLM)", W2["vllm-gpu-mc"], "85", "38.2", "71.9", "89", 38),
      ("30–45 160 GB", W2["sglang-host160-mc"], "100", "52.2", "61.3", "67", 39)]
for lab, s, pd_, ph, pm, pt, ln in T5:
    C(f"Table 5 {lab}: done", f + f":{ln}", pd_, done(s), "completed/requests, round 1")
    C(f"Table 5 {lab}: hit", f + f":{ln}", ph, 100 * s["cache_hit"], "cache_hit, round 1")
    C(f"Table 5 {lab}: mJ/tok", f + f":{ln}", pm, mjt(s), "j_per_prompt_token, round 1")
    C(f"Table 5 {lab}: TTFT", f + f":{ln}", pt, s["ttft_p50"], "ttft_p50, round 1")
C("f = 0.76 used on production traffic", f + ":52", "0.76", F_FIT, "fitted f")
C("busy card: right to within about 1%", f + ":54", "1", max(prod_model_err), "max |pred/meas - 1| over the five busy cells (both GPU rounds)", kind="le", about=True)
C("model: GPU cache w1 predicted 0.721", f + ":53", "0.721", prod_model[0][0], "1 - 0.76 h (r1)")
C("model: GPU cache w1 measured 0.717-0.718", f + ":53", "0.717–0.718", (min(prod_model[0][1], prod_model[1][1]), max(prod_model[0][1], prod_model[1][1])),
  "J/prompt token GPU / no cache, rounds 1 and 2 (stored j_per_prompt_token)")
for i, (pp, pm, ln) in enumerate((("0.606", "0.601", 54), ("0.584", "0.578", 54), ("0.709", "0.708", 54), ("0.603", "0.603", 54)), start=2):
    lab = ["160 GB w1", "300 GB w1", "GPU w2", "160 GB w2"][i - 2]
    C(f"model: {lab} predicted", f + f":{ln}", pp, prod_model[i][0], "1 - 0.76 h")
    C(f"model: {lab} measured", f + f":{ln}", pm, prod_model[i][1], "ratio to no cache, same window")
C("1/8: GPU cache power 544 W", f + ":55", "544", W2E["vllm-gpu-mc"]["avg_w"], "avg_w")
m18 = ratio(W2E["vllm-gpu-mc"], W2E["vllm-nocache-mc"])
C("1/8: measured 0.847", f + ":55", "0.847", m18, "J/prompt token ratio")
C("1/8: model 0.653", f + ":56", "0.653", 1 - F_P * W2E["vllm-gpu-mc"]["cache_hit"], "1 - 0.76 h")
C("1/8: implied f 0.33", f + ":57", "0.33", (1 - m18) / W2E["vllm-gpu-mc"]["cache_hit"], "(1 - ratio)/h")
C("1/8: median TTFT 1.2 s", f + ":58", "1.2", W2E["vllm-gpu-mc"]["ttft_p50"], "GPU cache")
C("1/8: median TTFT 18 s no cache", f + ":58", "18", W2E["vllm-nocache-mc"]["ttft_p50"], "no cache")
C("1/8: p90 7.1 s", f + ":58", "7.1", W2E["vllm-gpu-mc"]["ttft_p90"], "GPU cache")
C("1/8: p90 77 s no cache", f + ":58", "77", W2E["vllm-nocache-mc"]["ttft_p90"], "no cache")
C("predictor within 0.6 points at 1/8", f + ":63", "0.6", max(keepup_hit_err[:2]), "w2 1/8 GPU and 160 GB", kind="le")
C("predictor 2.3-4.3 points at 1/4", f + ":63", "2.3–4.3", (min(keepup_hit_err[2:]), max(keepup_hit_err[2:])),
  "w1 160/300 GB, w2 160 GB at 1/4", tol=0.05, note="300 GB prediction from the plan (0.525): +-0.05 pp input tolerance")
C("45.65% measured (GPU, 1/8)", f + ":63", "45.65", 100 * W2E["vllm-gpu-mc"]["cache_hit"], "cache_hit")
C("46.25% predicted", f + ":63", "46.25", 100 * PRED2E["244928"], "lru_hit_pred[244928]")
C("52.62% measured (160 GB, 1/8)", f + ":64", "52.62", 100 * W2E["sglang-host160-mc"]["cache_hit"], "cache_hit")
C("52.66% predicted", f + ":64", "52.66", 100 * PRED2E["610351"], "lru_hit_pred[610351]")
C("51.8% measured (160 GB, 1/4 w1)", f + ":64", "51.8", 100 * W1["sglang-host160-mc"]["cache_hit"], "cache_hit")
C("54.8% measured (300 GB)", f + ":64", "54.8", 100 * W1["sglang-host300-mc"]["cache_hit"], "cache_hit")
C("47.5% predicted (160 GB)", f + ":64", "47.5", 100 * PRED1["610351"], "lru_hit_pred[610351]")
C("overloaded: 36.7% measured", f + ":65", "36.7", 100 * W1["vllm-gpu-mc"]["cache_hit"], "w1 GPU")
C("overloaded: 41.8% predicted", f + ":65", "41.8", 100 * PRED1["244928"], "lru_hit_pred")
C("overloaded: 38.2% measured", f + ":65", "38.2", 100 * W2["vllm-gpu-mc"]["cache_hit"], "w2 GPU")
C("overloaded: 44.7% predicted", f + ":65", "44.7", 100 * PRED2Q["244928"], "lru_hit_pred")
C("chained prediction within 4-6%", f + ":71", "4–6", (min(chain), max(chain)), "(1 - 0.76 h_pred) vs measured ratio, w1 160 and 300 GB",
  tol=100 * F_P * 0.0005 / 0.578, note="300 GB uses the plan's 0.525 prediction (input tolerance propagated)")
# Figure 5 caption (adaptive; the figure is in results_control.tex)
ag = [abs(chg(*[s["j_per_turn"] for s in AD[k][w]])) for k in AD for w in ("gs", "as_")]
C("Fig. 5: two rounds agree within 1.5%", "results_control.tex:58", "1.5", max(ag), "|r1/r2 - 1| J/turn, GPU and adaptive, every cell", kind="le")
bl = [abs(chg(AD[k]["aJ"], AD[k]["gJ"])) for k in (("8b", 64), ("lc", 4), ("lc", 8))]
C("Fig. 5: below the cliff within 3%", "results_control.tex:60", "3", max(bl), "8B/64, 27k/4, 27k/8", kind="le")
C("Fig. 5: above it 41-45%", "results_control.tex:61", "41–45", (min(ad_save), max(ad_save)), "means of rounds")
tail = [mean(AD[k]["gp"]) / mean(AD[k]["ap"]) for k in (("8b", 128), ("8b", 256), ("lc", 16), ("lc", 32))]
C("Fig. 5: latency tail cut 4.5-123x", "results_control.tex:61", "4.5–123", (min(tail), max(tail)), "mean GPU p90 / mean adaptive p90 (as plotted)")

# ======================================================================================================================
# SECTION 8: admission control (Tables 6, 7)
# ======================================================================================================================
f = "results_control.tex"
gv = {(N_, P): {rd: sel(A, "vllm-gpu", N_, P, rnd=rd)[0]["j_per_turn"] for rd in (1, 2)} for N_ in (48, 96) for P in (5.0, 30.0)}
gate = {(v, N_, P): {rd: sel(ADMIT, v, N_, P, rnd=rd)[0]["j_per_turn"] for rd in (1, 2)} for v in ("vllm-admit16x4", "vllm-admit16x8") for N_ in (48, 96) for P in (5.0, 30.0)}
gchg = lambda N_, P: [chg(gate[(v, N_, P)][rd], gv[(N_, P)][rd]) for v in ("vllm-admit16x4", "vllm-admit16x8") for rd in (1, 2)]
s5 = [-x for N_ in (48, 96) for x in gchg(N_, 5.0)]
C("static gate cut 41-47% (5 s)", f + ":10", "41–47", (min(s5), max(s5)), "round-matched vs vLLM GPU cache, q = 4 and 8, 48/96 sessions")
C("static gate +6-8% at 48 (30 s)", f + ":10", "6–8", (min(gchg(48, 30.0)), max(gchg(48, 30.0))), "round-matched")
C("static gate +11-16% at 96 (30 s)", f + ":10", "11–16", (min(gchg(96, 30.0)), max(gchg(96, 30.0))), "round-matched")
ns48 = [48 - served(s) for v in ("vllm-admit16x4", "vllm-admit16x8") for P in (5.0, 30.0) for s in sel(ADMIT, v, 48, P)]
ns96 = [96 - served(s) for v in ("vllm-admit16x4", "vllm-admit16x8") for P in (5.0, 30.0) for s in sel(ADMIT, v, 96, P)]
C("never served 6-32 of 48", f + ":11", "6–32", (min(ns48), max(ns48)), "sessions with no turn in the step")
C("never served 53-80 of 96", f + ":11", "53–80", (min(ns96), max(ns96)), "sessions with no turn")
C("10-min GPU cache: 333 turns", f + ":12", "333", M(LONG, "vllm-gpu-long", "turns", 48), "mean of rounds")
C("10-min GPU cache: 1,185 J", f + ":13", "1,185", M(LONG, "vllm-gpu-long", "j_per_turn", 48), "mean of rounds")
C("10-min GPU cache: median resume 54 s", f + ":13", "54", M(LONG, "vllm-gpu-long", "resume_latency_p50", 48), "mean of rounds")
C("10-min GPU cache: p90 70 s", f + ":13", "70", M(LONG, "vllm-gpu-long", "resume_latency_p90", 48), "mean of rounds")
C("10-min static gate: 456 turns", f + ":13", "456", M(LONG, "vllm-admit16x4-long", "turns", 48), "mean of rounds")
C("10-min static gate: 761 J", f + ":13", "761", M(LONG, "vllm-admit16x4-long", "j_per_turn", 48), "mean of rounds")
C("10-min static gate: median 1.4 s", f + ":13", "1.4", M(LONG, "vllm-admit16x4-long", "resume_latency_p50", 48), "mean of rounds")
C("10-min static gate: p90 about 146 s", f + ":14", "146", M(LONG, "vllm-admit16x4-long", "resume_latency_p90", 48), "mean of rounds")
B("latency tail doubled", f + ":12", "333", f"{M(LONG,'vllm-admit16x4-long','resume_latency_p90',48)/M(LONG,'vllm-gpu-long','resume_latency_p90',48):.2f}x",
  round(M(LONG, "vllm-admit16x4-long", "resume_latency_p90", 48) / M(LONG, "vllm-gpu-long", "resume_latency_p90", 48)) == 2, "p90 ratio rounds to 2")
dts = {round(b["t"] - a["t"], 3) for s in sel(CTRL, "vllm-formula") for a, b in zip(s["controller"]["trace"], s["controller"]["trace"][1:])}
B("K recomputed every 2 s", f + ":17", "2", str(sorted(dts)), dts == {2.0}, "controller trace interval")
fb = lambda N_, P: chg(M(CTRL, "vllm-formula", "j_per_turn", N_, P), M(CTRL, "vllm-gpu-base", "j_per_turn", N_, P))
for N_, P, pr, ln in ((48, 5.0, "40", 17), (96, 5.0, "31", 17), (48, 30.0, "7", 17), (96, 30.0, "1", 17)):
    C(f"formula saved at {N_}/{P:g} s", f + f":{ln}", pr, -fb(N_, P), "means vs vLLM GPU cache of the same run")
C("formula served 20 of 48", f + ":18", "20", mean(served(s) for s in sel(CTRL, "vllm-formula", 48)), "sessions with >= 1 turn (5 s)")
C("formula served 21 of 96", f + ":18", "21", mean(served(s) for s in sel(CTRL, "vllm-formula", 96)), "sessions with >= 1 turn")
C("10-min formula: 500 turns", f + ":19", "500", M(LONG, "vllm-formula-long", "turns", 48), "mean of rounds")
C("10-min formula: 682 J", f + ":19", "682", M(LONG, "vllm-formula-long", "j_per_turn", 48), "mean of rounds")
C("10-min formula: median 0.5 s", f + ":19", "0.5", M(LONG, "vllm-formula-long", "resume_latency_p50", 48), "mean of rounds")
C("10-min formula: p90 about 147 s", f + ":19", "147", M(LONG, "vllm-formula-long", "resume_latency_p90", 48), "mean of rounds")
aimd = [-chg(M(CTRL, "vllm-aimd", "j_per_turn", N_, P), M(CTRL, "vllm-gpu-base", "j_per_turn", N_, P)) for N_ in (48, 96) for P in (5.0, 30.0)]
C("AIMD saved at most 1.3%", f + ":20", "1.3", max(aimd), "max saving over 4 synthetic cells", kind="le")
C("AIMD real: saved 5% at 96", f + ":21", "5", -chg(M(REAL, "vllm-aimd-real", "j_per_turn", 96), M(REAL, "vllm-gpu-real", "j_per_turn", 96)), "means")
C("AIMD real: cost 6% more at 48", f + ":21", "6", chg(M(REAL, "vllm-aimd-real", "j_per_turn", 48), M(REAL, "vllm-gpu-real", "j_per_turn", 48)), "means")
aimd_max60 = {max(x["slots"] for x in s["controller"]["trace"] if x["t"] <= 60) for N_ in (48, 96) for P in (5.0, 30.0) for s in sel(CTRL, "vllm-aimd", N_, P)}
C("AIMD window rose to 44 in first minute", f + ":22", "44", max(aimd_max60), "max slots with t <= 60 s, every step")
C("AIMD window fell to 1", f + ":22", "1", min(x["slots"] for N_ in (48, 96) for P in (5.0, 30.0) for s in sel(CTRL, "vllm-aimd", N_, P) for x in s["controller"]["trace"]), "min slots")
realtr = [[x["slots"] for x in s["controller"]["trace"]] for N_ in (16, 48, 96) for s in sel(REAL, "vllm-aimd-real", N_)]
C("AIMD real: only grew, to 158", f + ":23", "158", max(max(t) for t in realtr), "max slots") if all(all(b >= a for a, b in zip(t, t[1:])) for t in realtr) else \
    B("AIMD real: only grew", f + ":23", "158", "not monotone", False, "")
f1 = [chg(M(REALHI, "vllm-formula-real-hi", "j_per_turn", N_), M(REALHI, "vllm-gpu-real-hi", "j_per_turn", N_)) for N_ in (192, 288)]
C("first formula +31% at 192/288", f + ":28", "31", mean(f1), "means (%.1f, %.1f)" % tuple(f1))
ft = [-chg(M(REALHI, "vllm-formula-real-hi", "turns", N_), M(REALHI, "vllm-gpu-real-hi", "turns", N_)) for N_ in (192, 288)]
C("first formula served 40-50% fewer turns", f + ":29", "40–50", (min(ft), max(ft)), "turns")
sp = {N_: (M(P5REAL, "vllm-formula-mean-real", "j_per_turn", N_), M(REALHI, "vllm-gpu-real-hi", "j_per_turn", N_)) for N_ in (192, 288)}
C("shared-prefix formula -13% at 192", f + ":30", "−13", chg(*sp[192]), "means")
C("shared-prefix formula -15% at 288", f + ":30", "−15", chg(*sp[288]), "means")
for (N_, i), pr in zip(((192, 0), (288, 0), (192, 1), (288, 1)), ("80.2", "73.0", "92.4", "85.6")):
    C(f"shared-prefix J/turn {('formula', 'GPU')[i]} {N_}", f + ":30", pr, sp[N_][i], "mean of rounds")
hd_ = [100 * (M(P5REAL, "vllm-formula-mean-real", "cache_hit", N_) - M(REALHI, "vllm-gpu-real-hi", "cache_hit", N_)) for N_ in (192, 288)]
C("hits 8.5-9.6 points higher", f + ":30", "8.5–9.6", (min(hd_), max(hd_)), "means")
for N_, pr in ((192, "1.5"), (288, "8")):
    C(f"shared-prefix fewer turns at {N_}", f + ":31", pr, -chg(M(P5REAL, "vllm-formula-mean-real", "turns", N_), M(REALHI, "vllm-gpu-real-hi", "turns", N_)), "means")
rr2 = [abs(chg(a, b)) for N_ in (192, 288) for a, b in [V(P5REAL, "vllm-formula-mean-real", "j_per_turn", N_)]]
C("repeated within 1.5% in a second round", f + ":31", "1.5", max(rr2), "|r1/r2 - 1| J/turn", kind="le")
moe_f = [chg(M(MOEREAL, "vllm-formula-moe-real", "j_per_turn", N_), M(MOEREAL, "vllm-gpu-moe-real", "j_per_turn", N_)) for N_ in (32, 96, 192)]
C("MoE: shared-prefix formula +0.3% to -2.5%", f + ":32", "+0.3% to −2.5", (min(moe_f), max(moe_f)), "means, 32/96/192 sessions")
C("MoE 192: GPU p90 0.37 s", f + ":33", "0.37", M(MOEREAL, "vllm-gpu-moe-real", "resume_latency_p90", 192), "mean of rounds")
C("MoE 192: formula p90 0.22 s", f + ":33", "0.22", M(MOEREAL, "vllm-formula-moe-real", "resume_latency_p90", 192), "mean of rounds")
th = V(REALHI, "sglang-host160-real-hi", "cache_hit", 192) + V(REALHI, "sglang-host160-real-hi", "cache_hit", 288)
C("tier alone held 89-91% (real)", f + ":36", "89–91", (100 * min(th), 100 * max(th)), "per round, 192/288")
gt = [chg(M(P5REAL, "sglang-host160-formula-real", "j_per_turn", N_), M(REALHI, "sglang-host160-real-hi", "j_per_turn", N_)) for N_ in (192, 288)]
C("gate + tier: +10% at 192", f + ":36", "10", gt[0], "means")
C("gate + tier: nothing at 288", f + ":36", "0", gt[1], "means (rounds to 0%)")
gtt = [-chg(M(P5REAL, "sglang-host160-formula-real", "turns", N_), M(REALHI, "sglang-host160-real-hi", "turns", N_)) for N_ in (192, 288)]
C("gate + tier: 15-18% fewer turns", f + ":37", "15–18", (min(gtt), max(gtt)), "means")
capg = {s["controller"]["capacity"] for N_ in (192, 288) for s in sel(P5REAL, "sglang-host160-formula-real", N_)}
C("gate sized to GPU's 251k", f + ":38", "251", max(capg) / 1000, "controller capacity recorded (k)", about=True)
C("tier's 610k", f + ":38", "610", CAP["host160"] / 1000, "SGLang host pool (k)", about=True)
slots = [x["slots"] for N_ in (192, 288) for s in sel(P5REAL, "sglang-host160-formula-real", N_) for x in s["controller"]["trace"]]
C("admitted a median of about 100", f + ":40", "100", statistics.median(slots), "median slots over all control intervals (per-step medians %s)"
  % ", ".join(str(statistics.median([x['slots'] for x in s['controller']['trace']])) for N_ in (192, 288) for s in sel(P5REAL, "sglang-host160-formula-real", N_)), about=True)
C("admitted 24-280 over a step", f + ":40", "24–280", (min(slots), max(slots)), "min/max slots")
C("synthetic: tier alone 35% hits at 96 (10 min)", f + ":41", "35", 100 * M(FAIR, "sglang-host160-fair", "cache_hit", 96), "mean of rounds")
C("gate + tier: -36%", f + ":41", "36", -chg(M(FAIR, "sglang-host160-formula-fair", "j_per_turn", 96), M(FAIR, "sglang-host160-fair", "j_per_turn", 96)), "means")
C("gate + tier: 34.5% more turns", f + ":41", "34.5", chg(M(FAIR, "sglang-host160-formula-fair", "turns", 96), M(FAIR, "sglang-host160-fair", "turns", 96)), "means")
B("gate + tier served all 96", f + ":42", "96", str([served(s) for s in sel(FAIR, "sglang-host160-formula-fair", 96)]),
  all(served(s) == 96 for s in sel(FAIR, "sglang-host160-formula-fair", 96)), "sessions with >= 1 turn")
C("median resume latency from 110 s", f + ":42", "110", M(FAIR, "sglang-host160-fair", "resume_latency_p50", 96), "mean of rounds")
C("median resume latency to 16 s", f + ":42", "16", M(FAIR, "sglang-host160-formula-fair", "resume_latency_p50", 96), "mean of rounds")
C("longer tail 204 s", f + ":42", "204", M(FAIR, "sglang-host160-formula-fair", "resume_latency_p90", 96), "mean of rounds")
C("against 124 s", f + ":43", "124", M(FAIR, "sglang-host160-fair", "resume_latency_p90", 96), "mean of rounds")
gate_lc_ok = all(x["gate_on"] == (x["w"] > 1.2 * s["controller"]["capacity"]) for N_ in (4, 8, 16, 32) for s in sel(LC, "vllm-adaptive-lc", N_) for x in s["controller"]["trace"])
B("adaptive gate applies K only above theta C, theta = 1.2", f + ":46", "1.2", "gate_on == (w > 1.2 C) in every control interval" if gate_trace_ok and gate_lc_ok else "mismatch",
  gate_trace_ok and gate_lc_ok, "controller traces of all adaptive runs (8B and 27k, both rounds)")
dt8 = {round(b["t"] - a["t"], 3) for s in sel(AD8, "vllm-adaptive-8b") for a, b in zip(s["controller"]["trace"], s["controller"]["trace"][1:])}
B("2 s control intervals", f + ":49", "2", f"{min(dt8):.1f}-{max(dt8):.1f} s", all(abs(x - 2) < 0.5 for x in dt8), "adaptive controller trace intervals round to 2 s")
lcd = [abs(chg(AD[("lc", N_)]["aJ"], AD[("lc", N_)]["gJ"])) for N_ in (4, 8)]
C("adaptive within 0.2% at 4 and 8 long-context", f + ":48", "0.2", max(lcd), "means", kind="le")
gon = [sum(x["gate_on"] for x in s["controller"]["trace"]) / len(s["controller"]["trace"]) for s in sel(AD8, "vllm-adaptive-8b", 64)]
B("gate on in about a fifth of intervals (8B/64)", f + ":48", "64", ", ".join(f"{x:.3f}" for x in gon), all(0.15 <= x <= 0.25 for x in gon), "share of 2 s intervals with gate_on; 'about a fifth' read as 0.15-0.25")
w64 = max(r["gate_wait_s"] for s in sel(AD8, "vllm-adaptive-8b", 64) for r in rows(s) if r["turn"] > 0)
C("resumed turns waited up to 17 s", f + ":49", "17", w64, "max gate wait of resumed turns, both rounds")
form_cost = [chg(M(LC, "vllm-formula-lc", "j_per_turn", N_), AD[("lc", N_)]["gJ"]) for N_ in (4, 8)] + [chg(M(B8, "vllm-formula-8b", "j_per_turn", 64), AD[("8b", 64)]["gJ"])]
C("formula alone cost up to 2.7% more", f + ":49", "2.7", max(form_cost), "below-cliff cells, means")
f8p = V(B8, "vllm-formula-8b", "resume_latency_p90", 64)
C("formula 8B/64 p90 8-12 s", f + ":50", "8–12", (min(f8p), max(f8p)), "per round")
C("GPU 8B/64 p90 1.6-1.9 s", f + ":50", "1.6–1.9", (min(AD[('8b', 64)]['gp']), max(AD[('8b', 64)]['gp'])), "per round")
C("adaptive 41% at 128 and 256 (8B)", f + ":51", "41", max(-chg(AD[("8b", N_)]["aJ"], AD[("8b", N_)]["gJ"]) for N_ in (128, 256)), "means (both round to 41)")
C("adaptive 41% at 128 (8B, other cell)", f + ":51", "41", min(-chg(AD[("8b", N_)]["aJ"], AD[("8b", N_)]["gJ"]) for N_ in (128, 256)), "means")
C("adaptive 44-45% at 16/32 long-context", f + ":51", "44–45", tuple(sorted(-chg(AD[("lc", N_)]["aJ"], AD[("lc", N_)]["gJ"]) for N_ in (16, 32))), "means")
g8 = AD[("8b", 128)]["gp"] + AD[("8b", 256)]["gp"]
a8 = AD[("8b", 128)]["ap"] + AD[("8b", 256)]["ap"]
gl = AD[("lc", 16)]["gp"] + AD[("lc", 32)]["gp"]
al = AD[("lc", 16)]["ap"] + AD[("lc", 32)]["ap"]
C("8B GPU p90 24-81 s", f + ":52", "24–81", (min(g8), max(g8)), "per round, 128/256")
C("8B adaptive p90 0.6-1.5 s", f + ":52", "0.6–1.5", (min(a8), max(a8)), "per round")
C("27k GPU p90 40-121 s", f + ":52", "40–121", (min(gl), max(gl)), "per round, 16/32")
C("27k adaptive p90 8.7-12.7 s", f + ":53", "8.7–12.7", (min(al), max(al)), "per round")
fpl = lambda k, w: [p_lo(first_lat(s)) for s in AD[k][w]]
C("first-turn p90 gate 8B/128: 88-91 s", f + ":56", "88–91", (min(fpl(("8b", 128), "as_")), max(fpl(("8b", 128), "as_"))), "per round, p90 = v[int(0.9n)-1] of TTFT+wait, served sessions")
C("first-turn p90 gate 8B/256: 130 s", f + ":56", "130", mean(fpl(("8b", 256), "as_")), "mean of rounds (%s)" % ", ".join(f"{x:.1f}" for x in fpl(("8b", 256), "as_")))
C("first-turn p90 GPU 8B/128: 38 s", f + ":57", "38", mean(fpl(("8b", 128), "gs")), "mean of rounds")
C("first-turn p90 GPU 8B/256: 88 s", f + ":57", "88", mean(fpl(("8b", 256), "gs")), "mean of rounds")
C("first-turn p90 gate 27k/16: 112-113 s", f + ":57", "112–113", (min(fpl(("lc", 16), "as_")), max(fpl(("lc", 16), "as_"))), "per round")
C("first-turn p90 gate 27k/32: 110-114 s", f + ":57", "110–114", (min(fpl(("lc", 32), "as_")), max(fpl(("lc", 32), "as_"))), "per round")
C("first-turn p90 GPU 27k/16: 62 s", f + ":57", "62", mean(fpl(("lc", 16), "gs")), "mean of rounds")
C("first-turn p90 GPU 27k/32: 132 s", f + ":58", "132", mean(fpl(("lc", 32), "gs")), "mean of rounds")
longer = sum(mean(fpl(k, "as_")) > mean(fpl(k, "gs")) for k in (("8b", 128), ("8b", 256), ("lc", 16), ("lc", 32)))
B("longer in three of the four cells", f + ":58", "130", f"{longer} of 4", longer == 3, "mean first-turn p90 gate > GPU")


def cens(step, dur=150.0):
    N_ = step["sessions"]
    st_ = min(0.25, dur / 20 / N_)
    sv = {r["session"] for r in rows(step)}
    return [dur - i * st_ for i in range(N_) if i not in sv]


cw = [x for k in (("8b", 256), ("lc", 32)) for s in AD[k]["as_"] for x in cens(s)]
C("never-served sessions each waited at least 143 s", f + ":59", "143", min(cw), "150 s - staggered start (i x min(0.25, 150/20/N)) of each never-served session", kind="ge")
withc = all(mean(p_lo(first_lat(s) + cens(s)) for s in AD[k]["as_"]) > mean(fpl(k, "gs")) for k in (("8b", 128), ("8b", 256), ("lc", 16), ("lc", 32)))
B("longer in all four once never-served are counted", f + ":59", "143", "yes" if withc else "no", withc, "first-turn p90 with censored waits")
C("gate served 170 of 256", f + ":59", "170", mean(served(s) for s in AD[("8b", 256)]["as_"]), "both rounds")
C("gate served 16-17 of 32", f + ":59", "16–17", (min(served(s) for s in AD[("lc", 32)]["as_"]), max(served(s) for s in AD[("lc", 32)]["as_"])), "per round")
al8 = [p_lo(all_lat(s)) for k in (("8b", 128), ("8b", 256)) for s in AD[k]["as_"]]
gl8 = [p_lo(all_lat(s)) for k in (("8b", 128), ("8b", 256)) for s in AD[k]["gs"]]
all_lc = [p_lo(all_lat(s)) for k in (("lc", 16), ("lc", 32)) for s in AD[k]["as_"]]
gll = [p_lo(all_lat(s)) for k in (("lc", 16), ("lc", 32)) for s in AD[k]["gs"]]
C("all-turn p90 gate 8B: 68-86 s", f + ":61", "68–86", (min(al8), max(al8)), "p90 = v[int(0.9n)-1], per round, 128/256")
C("all-turn p90 GPU 8B: 30-82 s", f + ":61", "30–82", (min(gl8), max(gl8)), "same")
C("all-turn p90 gate 27k: 77-82 s", f + ":62", "77–82", (min(all_lc), max(all_lc)), "same, 16/32",
  note="nearest-rank v[int(0.9n)] gives %.1f-%.1f" % (min(p_nr(all_lat(s)) for k in (('lc', 16), ('lc', 32)) for s in AD[k]['as_']), max(p_nr(all_lat(s)) for k in (('lc', 16), ('lc', 32)) for s in AD[k]['as_'])))
C("all-turn p90 GPU 27k: 54-122 s", f + ":62", "54–122", (min(gll), max(gll)), "same")
C("10-min baseline 292 J (8B)", f + ":66", "292", T10["8b-gpu"]["j_per_turn"], "one round")
C("150 s baseline 261 J (8B)", f + ":67", "261", AD[("8b", 256)]["gJ"], "mean of rounds")
C("10-min baseline 3,022 J (27k)", f + ":67", "3,022", T10["lc-gpu"]["j_per_turn"], "one round")
C("150 s baseline 2,901 J (27k)", f + ":67", "2,901", AD[("lc", 32)]["gJ"], "mean of rounds")
C("10-min: 48% more turns (8B)", f + ":68", "48", more_turns10[0], "turns")
C("10-min: 62% more turns (27k)", f + ":68", "62", more_turns10[1], "turns")
C("10-min: 42% less energy (8B)", f + ":68", "42", less_e10[0], "J/turn")
C("10-min: 50% less energy (27k)", f + ":68", "50", less_e10[1], "J/turn")
fewer = []
for g_, a_ in (("8b-gpu", "8b-gate"), ("lc-gpu", "lc-gate")):
    _, cg_ = min_turns(T10[g_]); _, ca_ = min_turns(T10[a_])
    fewer.append(sum(1 for k in cg_ if ca_.get(k, 0) < cg_[k]))
B("no session completed fewer turns than under GPU cache", f + ":68", "42", f"sessions with fewer turns: {fewer}", fewer == [0, 0], "per-session turn counts, gate vs GPU")
for k, pr, ln in (("8b-gate", "183", 69), ("8b-gpu", "68", 69), ("lc-gate", "253", 69), ("lc-gpu", "131", 70)):
    C(f"10-min first-turn p90 {k}", f + f":{ln}", pr, fp[k], "v[int(0.9n)-1] of TTFT + wait, all sessions served")
for k, pr, ln in (("8b-gate", "226", 70), ("8b-gpu", "102", 70), ("lc-gate", "251", 70), ("lc-gpu", "130", 70)):
    C(f"10-min resumed p90 {k}", f + f":{ln}", pr, T10[k]["resume_latency_p90"], "stored resume_latency_p90")
B("H28 latency clause (<= 1.5x) did not hold", f + ":71", "1.5", f"{min(r10):.2f}-{max(r10):.2f}x", min(r10) > 1.5, "first-turn p90 ratio > 1.5 in both cells")
# Table 6
T6 = [(("8b", 64), "176.6", "171.9", "1.6–1.9", "1.3–1.5", 87), (("8b", 128), "269.6", "159.6", "24", "1.2–1.5", 88),
      (("8b", 256), "261.3", "154.5", "81", "0.6–0.7", 89), (("lc", 4), "1,495", "1,494", "0.27–0.28", "0.26–0.28", 90),
      (("lc", 8), "1,095", "1,094", "0.35–0.36", "0.35–0.36", 91), (("lc", 16), "2,964", "1,658", "40", "8.7–9.2", 92),
      (("lc", 32), "2,901", "1,589", "121", "11.0–12.7", 93)]
for k, pg, pa, pgp, pap, ln in T6:
    lab = f"Table 6 {'Qwen3-8B' if k[0] == '8b' else '32B 27k'} N={k[1]}"
    C(lab + ": J GPU", f + f":{ln}", pg, AD[k]["gJ"], "mean of rounds")
    C(lab + ": J adaptive", f + f":{ln}", pa, AD[k]["aJ"], "mean of rounds")
    C(lab + ": p90 GPU", f + f":{ln}", pgp, (min(AD[k]["gp"]), max(AD[k]["gp"])) if "–" in pgp else mean(AD[k]["gp"]), "range over rounds" if "–" in pgp else "both rounds")
    C(lab + ": p90 adaptive", f + f":{ln}", pap, (min(AD[k]["ap"]), max(AD[k]["ap"])), "range over rounds")
# Table 7
T7 = [("8b-gpu", 256, "256", "1,436", "292", "5", "68", 110), ("8b-gate", 256, "256", "2,119", "168", "6", "183", 111),
      ("lc-gpu", 32, "32", "143", "3,022", "4", "131", 112), ("lc-gate", 32, "32", "232", "1,507", "4", "253", 113)]
for k, N_, ps, pt, pj, pm, pf, ln in T7:
    C(f"Table 7 {k}: served", f + f":{ln}", ps, served(T10[k]), "sessions with >= 1 turn")
    C(f"Table 7 {k}: turns", f + f":{ln}", pt, T10[k]["turns"], "one round")
    C(f"Table 7 {k}: J/turn", f + f":{ln}", pj, T10[k]["j_per_turn"], "one round")
    C(f"Table 7 {k}: min turns per session", f + f":{ln}", pm, min_turns(T10[k])[0], "fewest turns of any session")
    C(f"Table 7 {k}: p90 first", f + f":{ln}", pf, fp[k], "v[int(0.9n)-1]")

# ======================================================================================================================
# SECTION 9: BFCL (Table 8)
# ======================================================================================================================
f = "results_accuracy.tex"
for v, lab, ln, pc, pb, pl, pt, pe, pk in (("vllm-gpu-bfcl", "GPU cache", 17, "204", "113", "91", "64", "0.619", "330"),
                                           ("vllm-nocache-bfcl", "No cache", 18, "201", "120", "81", "129", "1.239", "162")):
    C(f"Table 8 {lab}: correct", f + f":{ln}", pc, corr[v], "score files")
    C(f"Table 8 {lab}: base", f + f":{ln}", pb, BS[v]["base"][0], "BFCL_v4_multi_turn_base_score.json")
    C(f"Table 8 {lab}: long-context", f + f":{ln}", pl, BS[v]["long_context"][0], "long_context score file")
    C(f"Table 8 {lab}: time (min)", f + f":{ln}", pt, BF[v]["wall_s"] / 60, "wall_s / 60")
    C(f"Table 8 {lab}: energy (kWh)", f + f":{ln}", pe, kwh[v], "energy_j / 3.6e6")
    C(f"Table 8 {lab}: correct per kWh", f + f":{ln}", pk, perkwh[v], "correct / kWh")
C("accuracy differed by 3 of 400", f + ":23", "3", corr["vllm-gpu-bfcl"] - corr["vllm-nocache-bfcl"], "")
catd = [abs(BS["vllm-gpu-bfcl"][c][0] - BS["vllm-nocache-bfcl"][c][0]) for c in ("base", "long_context")]
C("category differences 7-10 tasks", f + ":23", "7–10", (min(catd), max(catd)), "|base diff|, |long-context diff|")
B("opposite directions", f + ":24", "10", f"base {BS['vllm-gpu-bfcl']['base'][0]-BS['vllm-nocache-bfcl']['base'][0]:+d}, long {BS['vllm-gpu-bfcl']['long_context'][0]-BS['vllm-nocache-bfcl']['long_context'][0]:+d}",
  (BS["vllm-gpu-bfcl"]["base"][0] - BS["vllm-nocache-bfcl"]["base"][0]) * (BS["vllm-gpu-bfcl"]["long_context"][0] - BS["vllm-nocache-bfcl"]["long_context"][0]) < 0, "signs differ")
B("half the time and half the energy", f + ":26", "2.0", f"time {BF['vllm-gpu-bfcl']['wall_s']/BF['vllm-nocache-bfcl']['wall_s']:.3f}, energy {kwh['vllm-gpu-bfcl']/kwh['vllm-nocache-bfcl']:.3f}",
  round(BF["vllm-gpu-bfcl"]["wall_s"] / BF["vllm-nocache-bfcl"]["wall_s"], 1) == 0.5 and round(kwh["vllm-gpu-bfcl"] / kwh["vllm-nocache-bfcl"], 1) == 0.5, "ratios round to 0.5")
C("2.0x correct per kWh", f + ":26", "2.0", perkwh["vllm-gpu-bfcl"] / perkwh["vllm-nocache-bfcl"], "ratio")
C("energy per correct task fell by 51%", f + ":27", "51", -chg(kwh["vllm-gpu-bfcl"] / corr["vllm-gpu-bfcl"], kwh["vllm-nocache-bfcl"] / corr["vllm-nocache-bfcl"]), "kWh per correct task")
n400 = {v: len(re.findall(r"400 Bad Request", text("phase7-results.zip", f"{v}-bfcl-r1-server.log"))) for v in BS}
ie = [BS[v]["long_context"][2] for v in BS]
C("23-24 long-context inference errors", f + ":27", "23–24", (min(ie), max(ie)), "error_type multi_turn:inference_error in the long-context score files")
mmb = {r["command"][r["command"].index("--max-model-len") + 1] for r in BFCL["runs"] if "--max-model-len" in r["command"]}
B("BFCL engine context 40,960 tokens", f + ":28", "40,960", ",".join(sorted(mmb)), mmb == {"40960"}, "vLLM --max-model-len in bfcl.json")
B("rejected by the engine (HTTP 400)", f + ":28", "400", f"HTTP 400 responses in server logs: {n400['vllm-gpu-bfcl']} / {n400['vllm-nocache-bfcl']}",
  [n400[v] for v in BS] == ie, "count of '400 Bad Request' in each engine log equals the inference-error count (the context-length cause is not logged)")

# ======================================================================================================================
# DISCUSSION and CONCLUSION
# ======================================================================================================================
f = "discussion.tex"
Wb = WEIGHTS_GIB * 2 ** 30
proj = {}
for hbm, ln in ((141e9, 6), (192e9, 7), (288e9, 7)):
    Cp = (hbm * 0.9 - Wb) / KV32
    proj[hbm] = (Cp, 1.2 * Cp / 68000, 2.2 * Cp / 68000)
C("projection H200: 4-8 sessions", f + ":6", "4–8", proj[141e9][1:], "kappa C/s, C = (141 GB x 0.9 - weights)/b_KV, s = 68k")
C("projection 192 GB: 7-13", f + ":6", "7–13", proj[192e9][1:], "same")
C("projection 288 GB: 13-24", f + ":7", "13–24", proj[288e9][1:], "same")
C("H200 about 440 kWh/month at full power", f + ":12", "440", 0.6 * 24 * 365 / 12, "600 W (spec) x 730 h", about=True)
C("kappa between 1.2 and 2.2 (discussion)", f + ":4", "1.2–2.2", (cross[5.0][0], cross[30.0][2]), "as intro")
C("model supported at about 11k", f + ":8", "11", med11 / 1000, "as abstract", about=True)
C("... and 27k tokens", f + ":8", "27", med27 / 1000, "as abstract", about=True)
price = A["settings"]["price"]
C("about $44 a month at $0.10/kWh", f + ":12", "44", 0.6 * 24 * 365 / 12 * 0.10, "440 kWh x $0.10 (assumed price; 600 W spec)")
C("card rented at $3.71 an hour", f + ":13", "3.71", price, "agents.json settings.price ($/h)")
C("about $2,700 a month", f + ":13", "2,700", price * 24 * 365 / 12, "price x 730 h", about=True)
C("production: finishing 85-86%", f + ":13", "85–86", (min(gpu_done), max(gpu_done)), "as Section 7")
moreturns = [chg(M(REAL, "sglang-host160-real", "turns", 96), M(REAL, "sglang-gpu-real", "turns", 96))] + \
            [chg(M(REALHI, "sglang-host160-real-hi", "turns", N_), M(REALHI, "vllm-gpu-real-hi", "turns", N_)) for N_ in (192, 288)]
C("22-33% more recorded agent turns", f + ":13", "22–33", (min(moreturns), max(moreturns)), "160 GB tier vs GPU cache at 96/192/288 sessions")
C("limitations: 15-17% host-tier gain", f + ":28", "15–17", (min(tier_less_mj), max(tier_less_mj)), "as Section 7")
C("conclusion: below the cliff 38-64%", f + ":34", "38–64", (100 * (1 - max(below)), 100 * (1 - min(below))), "as Figure 2")
C("conclusion: 0.01-1.6%", f + ":35", "0.01–1.6", (min(thr_all), max(thr_all)), "as abstract")
C("conclusion: 16-18%", f + ":37", "16–18", (min(tier_more_tok), max(tier_more_tok)), "as abstract")
C("conclusion: 1/110", f + ":38", "110", ratio_mean[4000], "ratio of means", about=True)
C("conclusion: 1/220", f + ":38", "220", ratio_mean[16000], "ratio of means", about=True)
C("conclusion: 42-50% less energy", f + ":40", "42–50", (min(less_e10), max(less_e10)), "as abstract")
C("conclusion: 48-62% more turns", f + ":40", "48–62", (min(more_turns10), max(more_turns10)), "as abstract")
C("conclusion: effects estimated from logs to within 4-6%", f + ":41", "4–6", (min(abs(x) for x in chain), max(abs(x) for x in chain)),
  "|chained (block-LRU hit -> Eq. 5) energy-ratio error|, w1 160/300 GB, as Sec. 7",
  note="hit-rate part alone is within %.1f pp" % max(keepup_hit_err))

# ======================================================================================================================
# TABLE 9 (pre-registration outcomes)
# ======================================================================================================================
f = "prereg_table.tex"
p2 = {}
for r in P2["runs"]:
    for s in r["steps"]:
        p2.setdefault((r["variant"], s["workload"], s["rate"]), []).append(s["j_per_output_token"])
dd = [abs(chg(mean(p2[("sglang-default", w, q)]), mean(p2[("vllm-default", w, q)]))) for (v, w, q) in p2 if v == "vllm-default"]
dt2 = [abs(chg(mean(p2[("sglang-tuned", w, q)]), mean(p2[("vllm-tuned", w, q)]))) for (v, w, q) in p2 if v == "vllm-tuned"]
C("H1 defaults within 2.3%", f + ":11", "2.3", max(dd), "max |SGLang/vLLM - 1| J/output token, mean of rounds, every workload x rate", kind="le")
C("H1 tuned up to 5.7%", f + ":11", "5.7", max(dt2), "same, tuned variants")
C("H2 1 user 1.46-1.48x", f + ":12", "1.46–1.48", (min(pen1), max(pen1)), "as Section 4")
C("H2 64 users 10.1-10.6x", f + ":12", "10.1–10.6", (min(pen64), max(pen64)), "as Section 4")
v80 = [chg(x, SG32[(48, 5.0)]) for x in V(A, "sglang-offload", "j_per_turn", 48)]
C("H3 80 GB -10 to -9%", f + ":13", "−10 to −9", (min(v80), max(v80)), "per round vs GPU mean")
C("H5 0.5-3% where SGLang tiers saved most", f + ":15", "0.5–3", (min(hc[(t, n, p)] for t, ns in (("sgl160", (48, 96)), ("sgl80", (48,))) for n in ns for p in (5.0, 30.0)),
                                                                    max(hc[(t, n, p)] for t, ns in (("sgl160", (48, 96)), ("sgl80", (48,))) for n in ns for p in (5.0, 30.0))), "host bound")
C("H5 11-14%", f + ":15", "11–14", (min(set11), max(set11)), "host bound")
C("H5 up to 212%", f + ":15", "212", max(v70h), "host bound, vLLM 70 GiB")
C("H6 -41 to -47% at 5 s", f + ":16", "−41 to −47", tuple(-x for x in (min(s5), max(s5))), "as Section 8")
C("H6 +6 to +8% at 30 s, 48", f + ":16", "+6 to +8", (min(gchg(48, 30.0)), max(gchg(48, 30.0))), "as Section 8")
C("H7 LFU hit 0.21%", f + ":17", "0.21", 100 * M(EVICT, "sglang-lfu", "cache_hit", 48), "48/5 s, mean of rounds")
C("H7 within 0.2% of LRU", f + ":17", "0.2", max(ev), "as Section 5", kind="le")
C("H7 20-28% worse on real trajectories", f + ":17", "20–28", (min(evr), max(evr)), "as Section 5")
C("H8 halve at 1.2 (5 s)", f + ":18", "1.2", cross[5.0][0], "as Section 5")
C("H8 halve at 1.5 (30 s)", f + ":18", "1.5", cross[30.0][0], "as Section 5")
C("H8 vanish by 1.55", f + ":18", "1.55", cross[5.0][2], "as Section 5")
C("H8 vanish by 2.2", f + ":18", "2.2", cross[30.0][2], "as Section 5")
C("H9 -57% at 48", f + ":19", "−57", chg(M(H160, "sglang-offload160", "j_per_turn", 48), SG32[(48, 5.0)]), "means")
C("H9 -16% at 96", f + ":19", "−16", chg(M(H160, "sglang-offload160", "j_per_turn", 96), SG32[(96, 5.0)]), "means")
C("H10 1/110", f + ":20", "110", ratio_mean[4000], "ratio of means", about=True)
C("H10 1/220", f + ":20", "220", ratio_mean[16000], "ratio of means", about=True)
C("H11 -0.08 to -0.16%", f + ":21", "−0.08 to −0.16", (min(min(v) for v in idle_d.values()), max(max(v) for v in idle_d.values())), "both engines")
C("H12 204", f + ":22", "204", corr["vllm-gpu-bfcl"], "")
C("H12 201", f + ":22", "201", corr["vllm-nocache-bfcl"], "")
C("H12 -51%", f + ":22", "−51", chg(kwh["vllm-gpu-bfcl"] / corr["vllm-gpu-bfcl"], kwh["vllm-nocache-bfcl"] / corr["vllm-nocache-bfcl"]), "energy per correct task")
for (N_, P), pr in (((48, 5.0), "−40"), ((96, 5.0), "−31"), ((48, 30.0), "−7"), ((96, 30.0), "−1")):
    C(f"H13 formula {N_}/{P:g} s", f + ":23", pr, fb(N_, P), "means")
below_aimd = sum(M(CTRL, "vllm-formula", "j_per_turn", N_, P) < M(CTRL, "vllm-aimd", "j_per_turn", N_, P) for N_ in (48, 96) for P in (5.0, 30.0))
C("H13 below AIMD in 3 of 4 cells", f + ":23", "3", below_aimd, "count of cells formula J < AIMD J")
C("H13 AIMD 0.55% lower in the fourth", f + ":23", "0.55", -chg(M(CTRL, "vllm-aimd", "j_per_turn", 96, 30.0), M(CTRL, "vllm-formula", "j_per_turn", 96, 30.0)), "96/30 s")
C("H14 68% hit at 96", f + ":24", "68", 100 * M(REAL, "vllm-gpu-real", "cache_hit", 96), "mean of rounds")
C("H14 72-73% at 192-288", f + ":24", "72–73", (100 * min(hh), 100 * max(hh)), "per round")
fr96 = [chg(a, b) for a, b in zip(V(REAL, "vllm-formula-real", "j_per_turn", 96), V(REAL, "vllm-gpu-real", "j_per_turn", 96))]
C("H14 formula -12 to -13% at 96", f + ":24", "−12 to −13", (min(fr96), max(fr96)), "round-matched")
C("H14 formula +31% at 192-288", f + ":24", "+31", mean(f1), "means")
for N_, pr in ((192, "−27"), (288, "−31")):
    C(f"H14 160 GB tier at {N_}", f + ":24", pr, chg(M(REALHI, "sglang-host160-real-hi", "j_per_turn", N_), M(REALHI, "vllm-gpu-real-hi", "j_per_turn", N_)), "means")
B("H15 resends after 25 min fully cached", f + ":25", "25", f"max delay {ret_max/60:.0f} min", ret_ok and ret_max == 1500, "retention.json")
C("H17 120 GB hit 50%", f + ":27", "50", 100 * M(SIZING, "sglang-host120", "cache_hit", 48), "48 sessions")
C("H17 240 GB at 96 -29%", f + ":27", "−29", chg(M(SIZING, "sglang-host240", "j_per_turn", 96), SG32[(96, 5.0)]), "means")
C("H17 240 GB hit 75.5% (48)", f + ":27", "75.5", 100 * M(SIZING, "sglang-host240", "cache_hit", 48), "mean")
C("H17 240 GB hit 37.2% (96)", f + ":27", "37.2", 100 * M(SIZING, "sglang-host240", "cache_hit", 96), "mean")
fs = V(SIZING, "sglang-host160-formula", "cache_hit", 96) + V(FSRR, "sglang-host160-formula-rerun", "cache_hit", 96)
C("H17 full stack 75% hit (96)", f + ":27", "75", 100 * mean(fs), "160 GB + formula at 96, run + rerun")
C("H17 full stack 49 of 96 served", f + ":27", "49", mean(served(s) for s in sel(SIZING, "sglang-host160-formula", 96) + sel(FSRR, "sglang-host160-formula-rerun", 96)), "sessions with >= 1 turn")
h18 = [100 * x for x in V(A, "sglang-offload", "cache_hit", 48)]
C("H18 11.4-12.6% -> 36.5%: 80 GB WT hit", f + ":28", "11.4–12.6", (min(h18), max(h18)), "per-round range")
C("H18 36.5%", f + ":28", "36.5", 100 * M(SIZING, "sglang-host80-wb", "cache_hit", 48), "mean")
C("H18 -27%", f + ":28", "−27", chg(M(SIZING, "sglang-host80-wb", "j_per_turn", 48), SG32[(48, 5.0)]), "means")
B("H19 96/96 served", f + ":29", "96", str([served(s) for s in sel(FAIR, "sglang-host160-formula-fair", 96)]), all(served(s) == 96 for s in sel(FAIR, "sglang-host160-formula-fair", 96)), "")
C("H19 -36%", f + ":29", "−36", chg(M(FAIR, "sglang-host160-formula-fair", "j_per_turn", 96), M(FAIR, "sglang-host160-fair", "j_per_turn", 96)), "means")
C("H19 +34.5% turns", f + ":29", "34.5", chg(M(FAIR, "sglang-host160-formula-fair", "turns", 96), M(FAIR, "sglang-host160-fair", "turns", 96)), "means")
h20 = [chg(M(CHEAP, "sglang-host80wb-formula", "j_per_turn", N_), mean(V(SIZING, "sglang-host160-formula", "j_per_turn", N_) + V(FSRR, "sglang-host160-formula-rerun", "j_per_turn", N_))) for N_ in (48, 96)]
C("H20 +20 to +21%", f + ":30", "+20 to +21", (min(h20), max(h20)), "80 GB WB + gate vs 160 GB + gate (run + rerun), 48/96")
C("H20 gate capacity 420k", f + ":30", "420", max(s["controller"]["capacity"] for s in sel(CHEAP, "sglang-host80wb-formula", 48)) / 1000, "controller capacity (k)", about=True)
C("H21 +10%", f + ":31", "+10", gt[0], "as Section 8")
C("H21 0%", f + ":31", "0", gt[1], "as Section 8")
C("H22 gate -41%", f + ":32", "−41", chg(AD[("8b", 128)]["aJ"], AD[("8b", 128)]["gJ"]), "adaptive/formula vs GPU at 128 (both cells -41%)")
h22 = [chg(M(P58B, "sglang-host160-8b", "j_per_turn", N_), M(P58B, "vllm-gpu-8b", "j_per_turn", N_)) for N_ in (128, 256)]
C("H22 tier -20%", f + ":32", "−20", h22[0], "160 GB vs vLLM GPU cache, 128")
C("H22 tier +2%", f + ":32", "+2", h22[1], "160 GB vs vLLM GPU cache, 256")
w1_err = [abs(100 * W1[v]["cache_hit"] - 100 * p) for v, p in (("vllm-gpu-mc", PRED1["244928"]), ("sglang-host160-mc", PRED1["610351"]), ("sglang-host300-mc", PLAN_PRED_300))]
C("H23 2 of 3 cells within 5 points (w1)", f + ":33", "2", sum(x <= 5 for x in w1_err), "count, errors %s pp" % ", ".join(f"{x:.2f}" for x in w1_err))
C("H23 0.04-0.6 points at 1/8", f + ":33", "0.04–0.6", tuple(sorted(keepup_hit_err[:2])), "w2 1/8")
C("H24 no cache 61%", f + ":34", "61", done(W1["vllm-nocache-mc"]), "w1 r1")
C("H24 GPU cache 85-86%", f + ":34", "85–86", (min(gpu_done), max(gpu_done)), "as Section 7")
C("H25 0.717-0.718", f + ":35", "0.717–0.718", (min(prod_model[0][1], prod_model[1][1]), max(prod_model[0][1], prod_model[1][1])), "as Section 7")
tiers_err = [abs(100 * (p / m_ - 1)) for p, m_ in (prod_model[2], prod_model[3], prod_model[5])]
C("H25 tiers within 1%", f + ":35", "1", max(tiers_err), "160/300 GB w1, 160 GB w2", kind="le")
h26 = (M(LC, "vllm-gpu-lc", "cache_hit", 4) >= 0.85 and M(LC, "vllm-gpu-lc", "cache_hit", 8) >= 0.70 and M(LC, "vllm-gpu-lc", "cache_hit", 32) < 0.05
       and abs(chg(M(LC, "vllm-gpu-lc", "j_per_turn", 32), M(LC, "vllm-nocache-lc", "j_per_turn", 32))) <= 3)
B("H26 supported (all clauses)", f + ":36", "85", "all clauses hold" if h26 else "a clause fails", h26, "hit >= 85% at 4, >= 70% at 8, < 5% at 32, |dJ| <= 3%")
C("H28 100% served", f + ":38", "100", 100 * min(served(T10["8b-gate"]) / 256, served(T10["lc-gate"]) / 32), "both cells")
C("H28 -42%", f + ":38", "−42", -less_e10[0], "8B")
C("H28 -50%", f + ":38", "−50", -less_e10[1], "27k")
C("H28 2.7x", f + ":38", "2.7", r10[0], "first-turn p90 ratio 8B")
C("H28 1.9x", f + ":38", "1.9", r10[1], "first-turn p90 ratio 27k")

# ======================================================================================================================
# Figure 1 (architecture) and appendix model numbers
# ======================================================================================================================
f = "fig_arch.tex"
C("Fig. 1: GPU hit 0-0.3 mJ", f + ":43", "0–0.3", (min(hit_tr), max(hit_tr)), "Table 1", unit=0.1)
C("Fig. 1: restore 0.27-0.68 mJ", f + ":44", "0.27–0.68", (min(min(v) for v in rst.values()), max(max(v) for v in rst.values())), "Table 1")
C("Fig. 1: recompute 56-73 mJ", f + ":45", "56–73", (min(rec6[4000]), max(rec6[16000])), "Table 1")
C("Fig. 1: kappa 1.2-2.2", f + ":59", "1.2–2.2", (cross[5.0][0], cross[30.0][2]), "as intro")
f = "fig_workflow.tex"
C("Fig. workflow: ~11k sessions", f + ":15", "11", med11 / 1000, "as abstract", about=True)
C("Fig. workflow: ~27k sessions", f + ":15", "27", med27 / 1000, "as abstract", about=True)
B("Fig. workflow: 400 trajectories, 400 BFCL tasks", f + ":17", "400", "400 / 400",
  len(text("phase2-results.zip", "swe-traj-400.jsonl").splitlines()) == 400 and sum(BS["vllm-gpu-bfcl"][c][1] for c in BS["vllm-gpu-bfcl"]) == 400, "jsonl lines, score totals")
B("Fig. workflow: vLLM 0.33, SGLang 0.5.23", f + ":32", "0.33", ", ".join(sorted(vers)), vers == {"0.33.0", "0.5.23"}, "run version fields")
f = "appendix.tex"
C("appendix: 31.2 B parameters", f + ":19", "31.2", params / 1e9, "architecture shapes")
C("appendix: 2.10 MFLOP", f + ":20", "2.10", 2 * 2 * 64 * 128 * 64 / 1e6, "2 x 64 x 128 per layer x 2 x 64 layers")
C("appendix: 62.4 GFLOP per token", f + ":19", "62.4", 2 * params / 1e9, "2 x parameters")
C("appendix: b_KV 262,144", f + ":29", "262,144", KV32, "2 x 64 x 8 x 128 x 2")
C("appendix: 1 - 1/e = 0.63", f + ":24", "0.63", 1 - 1 / math.e, "")
C("appendix: factor 1.58", f + ":25", "1.58", 1 / (1 - 1 / math.e), "")
C("appendix: 1.56 at C/s = 23", f + ":26", "1.56", (1 + (23 - 1) / (1 - 1 / math.e)) / 23, "N*/(C/s) with N* = 1 + (C/s - 1)/(1 - 1/e)")
C("appendix: weights 65.5 GB", f + ":29", "65.5", Wb / 1e9, "vLLM log 'Model loading took 61.03 GiB'")
C("appendix: C about 234k at 141 GB", f + ":30", "234", proj[141e9][0] / 1000, "(141e9 x 0.9 - weights)/262,144", about=True)
C("appendix: engine reported 245k", f + ":30", "245", CV / 1000, "vLLM log", about=True)
C("appendix: 409k at 192 GB", f + ":30", "409", proj[192e9][0] / 1000, "same", about=True)
C("appendix: 739k at 288 GB", f + ":31", "739", proj[288e9][0] / 1000, "same", about=True)
C("appendix: 4-8 sessions", f + ":31", "4–8", proj[141e9][1:], "kappa = 1.2-2.2, s = 68k")
C("appendix: 7-13 sessions", f + ":31", "7–13", proj[192e9][1:], "same")
C("appendix: 13-24 sessions", f + ":31", "13–24", proj[288e9][1:], "same")

# ======================================================================================================================
# NOT FROM DATA
# ======================================================================================================================
for t, loc, cat, note in [
    ("Kimi trace: 23,608 requests; prompt median 6,346 / p90 16,810 tokens", "setup.tex:50", "plan record", "trace statistics, plan Phase 6 (trace file not in the result archives)"),
    ("Kimi trace: 23,608 requests (workflow figure)", "fig_workflow.tex:17", "plan record", "as above"),
    ("Kimi trace: 55% of prompt blocks reused with an unbounded cache", "results_production.tex:6", "plan record", "plan Phase 6"),
    ("Kimi trace: same-conversation return median 114 s, p90 420 s", "results_production.tex:7", "plan record", "plan Phase 6"),
    ("300 GB block-LRU hit prediction 52.5%", "results_production.tex:64", "plan record", "plan Phase 6 (not in mooncake-r1.json; used with +-0.05 pp tolerance)"),
    ("first host for the 10-minute runs could not start vLLM 0.33 (driver too old)", "setup.tex:74", "plan record", "plan Phase 9 scope change"),
    ("third registered H28 cell (Qwen3-8B, 128 sessions) dropped before any data", "results_control.tex:65", "plan record", "plan Phase 9"),
    ("estimated 24k tokens per session behind the 12-20 band", "results_cliff.tex:54", "plan record", "Phase 8 pre-registration; input of the band check"),
    ("predicted 'under 30%' hits for recorded trajectories at 192/288", "results_cliff.tex:93", "plan record", "H14 prediction"),
    ("SWE-agent trajectories' final contexts about 12k tokens", "results_cliff.tex:94", "plan record", "plan; needs a tokenizer over swe-traj-400.jsonl"),
    ("gate + tier: predicted a further 25% saving", "results_control.tex:37", "plan record", "H21 prediction"),
    ("BFCL: predicted accuracy within two points and >= 20% less energy per correct task", "results_accuracy.tex:27", "plan record", "H12 prediction"),
    ("H28 latency threshold 1.5x and other registered thresholds (Table 9 'Prediction' column, commit times)", "prereg_table.tex:11", "plan record", "plan commits"),
    ("28 predictions written before the runs", "fig_workflow.tex:31", "plan record", "count of plan hypotheses"),
    ("Six later starts failed (four OOM, one script bug, one flag check); four re-run", "setup.tex:73", "plan record", "run logs / plan narrative"),
    ("Mao and Zhou: host-side difference under 1% of GPU saving", "results_cost.tex:19", "cited", "mao2026offloading"),
    ("Mao and Zhou: moving a KV byte >= 400x cheaper than recomputing", "results_cost.tex:20", "cited", "mao2026offloading"),
    ("Mao and Zhou: host cost below 1% (tiers section)", "results_tiers.tex:64", "cited", "mao2026offloading"),
    ("Mao and Zhou: buffer sizes of 8-128 GiB", "intro.tex:15", "cited", "mao2026offloading"),
    ("H200 NVL dense BF16 peak 836 TFLOP/s", "model.tex:19", "cited", "nvidia2024h200; input of the 69% check"),
    ("Lu et al.: cliff factor 1.9", "results_cliff.tex:41", "cited", "lu2026metastability"),
    ("Luo et al.: GPU telemetry misses 41-45% of an agent server's energy", "discussion.tex:21", "cited", "luo2026where"),
    ("Wang et al.: 54-62% reuse in Alibaba traces", "results_production.tex:6", "cited", "wang2025kvcachewild"),
    ("He: cliff factor near 1; Lu et al.: 1.9", "model.tex:57", "cited", "he2026robot, lu2026metastability (also results_cliff.tex:40-41)"),
    ("GitHub Copilot traces: median 68k prompt tokens per call", "results_cliff.tex:95", "cited", "copilot2026wild; input of the discussion projections"),
    ("68k-token sessions for production coding agents", "discussion.tex:5", "cited", "copilot2026wild"),
    ("CONCUR's published AIMD parameters", "results_control.tex:20", "cited", "chen2026concur"),
    ("1 nJ per byte of host-memory traffic", "results_tiers.tex:61", "assumption", "host-cost bound input"),
    ("10 W per busy core", "results_tiers.tex:60", "assumption", "host-cost bound input"),
    ("PCIe Gen5 bandwidth B ~ 50 GB/s", "model.tex:25", "assumption", "Eq. 2 input"),
    ("GPU board draw 100-250 W during the copy", "model.tex:26", "assumption", "Eq. 2 input"),
    ("electricity price $0.10 per kWh", "discussion.tex:12", "assumption", "input of the $44 check"),
    ("gate headroom factor 0.9 (10% of free memory)", "model.tex:91", "assumption", "Eq. 6 design constant"),
    ("theta = 1.2 switch threshold (also verified against the controller trace)", "model.tex:96", "assumption", "design constant"),
    ("u = 0.9 memory fraction in the projections", "appendix.tex:30", "assumption", "projection input (matches vLLM's 0.90 flag)"),
    ("lognormal pauses: sigma = 0.8, capped at six times the median", "setup.tex:42", "spec", "workload generator settings (agent_energy.py)"),
    ("H200 NVL: 141 GB HBM, PCIe Gen5 x16; container could use up to 454 GiB of host memory", "setup.tex:18", "spec", "hardware"),
    ("NVML counter read every 0.1 s (architecture figure)", "fig_arch.tex:55", "spec", "meter"),
    ("141/192/288 GB cards in the projection", "appendix.tex:30", "spec", "projection inputs"),
    ("second host: driver 595", "setup.tex:20", "spec", "hardware/software"),
    ("NVML counter read every 0.1 s; four warm-up requests; < 1 C over 60 s; 30 s idle", "setup.tex:31", "spec", "meter/protocol settings (settings.idle=30, settle_window=60 in the JSONs)"),
    ("512-token prompt blocks in the Kimi trace", "setup.tex:51", "spec", "trace format"),
    ("Qwen3-32B shapes (hidden 5,120, 64 layers, 64/8 heads of 128, MLP 25,600); 30B-A3B 48 layers/4 KV heads; 8B 36 layers", "appendix.tex:18", "spec", "model cards; inputs of the KiB/token and FLOP checks"),
    ("192 GB and 288 GB cards; H200 full power ~600 W", "discussion.tex:7", "spec", "inputs of the projections and the 440 kWh check"),
    ("vLLM 70 GiB tier size (container limit)", "results_tiers.tex:34", "spec", "run configuration"),
    ("Eq. 1-6 values (62.4 GFLOP, 2.10 MFLOP, 0.57 PFLOP/s, 68%, 1.03/0.87 pJ, 59/63/73 mJ, 5.2 us, 0.5-1.3 mJ, 45-140, 1.58, f = 0.76, 234k/409k/739k, 4-8/7-13/13-24, 440 kWh, $44)", "model.tex:21", "model calculation", "recomputed from their stated inputs and counted among the checks above"),
]:
    N(t, loc, cat, note)

# ======================================================================================================================
# Report
# ======================================================================================================================
def main():
    w = max(len(r["label"]) for r in RESULTS)
    print("VERIFY NUMBERS: paper/tex against results/*.zip\n")
    for r in RESULTS:
        st_ = "PASS" if r["ok"] else "FAIL"
        print(f"{st_}  {r['label']:<{w}}  {r['loc']:<28} paper {r['printed']:<16} recomputed {r['comp']:<22} [{r['rule']}]" + (f"  NOTE: {r['note']}" if r["note"] else ""))
    fails = [r for r in RESULTS if not r["ok"]]
    print("\nFAILURES")
    for r in fails:
        print(f"  {r['label']} ({r['loc']}): paper {r['printed']}, recomputed {r['comp']}" + (f"; {r['note']}" if r["note"] else ""))
    if not fails:
        print("  none")
    print("\nNOT FROM DATA")
    for cat in ("cited", "plan record", "assumption", "spec", "model calculation"):
        items = [x for x in NFD if x["cat"] == cat]
        if items:
            print(f"  [{cat}]")
            for x in items:
                print(f"    {x['loc']:<28} {x['text']}" + (f"  ({x['note']})" if x["note"] else ""))
    n, npass = len(RESULTS), sum(r["ok"] for r in RESULTS)
    print(f"\nSUMMARY: {n} checked, {npass} pass, {n - npass} fail, {len(NFD)} not-from-data")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
