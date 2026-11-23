"""Figures for the paper, read directly from the result archives in ../results (no hand-copied numbers).

Each value is the mean over the rounds present in the archive for that cell. The script also prints every plotted
value so it can be checked against the tables in tex/. Sizes are the printed sizes (column 3.375 in, text 7.0 in), so
8 pt text in a figure prints at 8 pt next to the 10 pt body text.
"""
import json, os, statistics, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figures") + "/"
INK, INK2, RULE = "#1a1a1a", "#52514e", "#d9d8d4"
COL = {"none": "#9a9893", "gpu": "#2a78d6", "gate": "#eb6834", "t160": "#1baf7a", "t300": "#127a55"}
COLW, TEXTW = 3.375, 7.0
plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ecebe8", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def load(archive, name):
    z = zipfile.ZipFile(os.path.join(RES, archive))
    return json.loads(z.read(next(n for n in z.namelist() if n.endswith(name))))


def cell(R, variant, key, sessions=None, frac=None, pause=5.0):
    """Mean of `key` over all rounds of `variant` at the given sessions (agent runs) or fraction (production)."""
    vals = []
    for r in R["runs"]:
        if r["variant"] != variant:
            continue
        for s in r.get("steps", []):
            if sessions is not None and (s.get("sessions") != sessions or s.get("pause_median_s", pause) != pause):
                continue
            if frac is not None and s.get("fraction") != frac:
                continue
            vals.append(s[key])
    assert vals, (variant, key, sessions, frac)
    return statistics.mean(vals)


def rounds(R, variant, key, sessions, pause=5.0):
    """All per-round values of `key` for one cell (the figures plot their mean)."""
    v = [s[key] for r in R["runs"] if r["variant"] == variant for s in r.get("steps", [])
         if s.get("sessions") == sessions and s.get("pause_median_s", pause) == pause]
    assert v, (variant, key, sessions)
    return v


# ---------- Figure: the memory wall. x = W/C (working set over GPU KV capacity), y = GPU-cache energy per turn over
# no-cache energy per turn. W sums each session's largest prompt in the step, as in Section 5.
r3 = load("rest-results.zip", "agents-p5-r3.json")
b8 = load("rest-results.zip", "agents-p5-8b.json")
moe = load("phase3-results.zip", "agents-moe.json")
lc1, lc2 = load("phase8-results.zip", "agents-longctx-r1.json"), load("phase8-results.zip", "agents-longctx-r2.json")
lc = {"runs": lc1["runs"] + lc2["runs"]}
CAP = {"32b": 244928, "8b": 777872, "moe": 704032}


def wc(R, variant, sessions, cap, pause=5.0):
    """Mean over rounds of W/C for one cell."""
    vals = []
    for r in R["runs"]:
        if r["variant"] != variant:
            continue
        for s in r.get("steps", []):
            if s.get("sessions") != sessions or s.get("pause_median_s", pause) != pause or not s.get("turns_detail"):
                continue
            F, mx = s["turns_detail"]["fields"], {}
            for row in s["turns_detail"]["rows"]:
                d = dict(zip(F, row))
                mx[d["session"]] = max(mx.get(d["session"], 0), d["prompt_tokens"] or 0)
            vals.append(sum(mx.values()) / cap)
    assert vals, (variant, sessions)
    return statistics.mean(vals)


# 30B-A3B is drawn first, larger and underneath: at W/C ~ 1.75 its 96-session point coincides with the 27k
# 16-session point, and the larger hollow diamond stays visible around the square.
series = [
    ("Qwen3-32B, 11k", r3, "vllm-gpu-r3", "vllm-nocache-r3", [16, 48, 96], "32b", dict(marker="o", color=INK, mfc=INK, ms=4.2, zorder=4)),
    ("Qwen3-32B, 27k", lc, "vllm-gpu-lc", "vllm-nocache-lc", [4, 8, 16, 32], "32b", dict(marker="s", color=INK, mfc="white", ms=4.2, zorder=4)),
    ("Qwen3-8B", b8, "vllm-gpu-8b", "vllm-nocache-8b", [64, 128, 256], "8b", dict(marker="^", color=INK2, mfc=INK2, ms=4.2, zorder=4)),
    ("Qwen3-30B-A3B", moe, "vllm-gpu-moe", "vllm-nocache-moe", [32, 96, 192], "moe", dict(marker="D", color=INK2, mfc="white", ms=6.2, zorder=3)),
]
fig, ax = plt.subplots(figsize=(COLW, 2.25))
ax.axvspan(1.2, 2.2, color="#f3f2ef", lw=0, zorder=0)
ax.text(1.7, 0.06, "cliff band\n$W/C$ = 1.2–2.2", ha="center", va="bottom", fontsize=7, color=INK2)
ax.axhline(1.0, color=INK2, lw=0.6, zorder=1)
ax.text(4.45, 1.015, "no cache", ha="right", va="bottom", fontsize=7, color=INK2)
print("wall: model, sessions, W/C, GPU J/turn, no-cache J/turn, ratio")
for name, R, g, n, xs, cap, st in series:
    X, Y = [], []
    for x in xs:
        gj, nj = cell(R, g, "j_per_turn", sessions=x), cell(R, n, "j_per_turn", sessions=x)
        w = wc(R, g, x, CAP[cap])
        X.append(w); Y.append(gj / nj)
        print(f"  {name:16s} {x:4d} {w:5.2f} {gj:8.1f} {nj:8.1f} {gj / nj:.3f}")
    ax.plot(X, Y, ls="none", mew=0.8, label=name, **st)
ax.set_xlim(0, 4.5); ax.set_ylim(0, 1.12)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1)); ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.set_xlabel("working set / GPU KV capacity ($W/C$)")
ax.set_ylabel("energy per turn, GPU cache / no cache")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42), handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.3); fig.savefig(OUT + "wall.pdf"); plt.close(fig)

# ---------- Figure: production replay, first window, 1/4 sample, round 1 (the round of Table 5; 300 GB ran once)
prod = load("phase6-results.zip", "mooncake-r1.json")
pol = [("no cache", "vllm-nocache-mc", "none"), ("GPU cache", "vllm-gpu-mc", "gpu"),
       ("160 GB tier", "sglang-host160-mc", "t160"), ("300 GB tier", "sglang-host300-mc", "t300")]
done, jt = [], []
print("production: policy, done %, mJ per prompt token")
for lab, v, _ in pol:
    d = 100 * cell(prod, v, "completed", frac=0.25) / cell(prod, v, "requests", frac=0.25)
    j = 1000 * cell(prod, v, "j_per_prompt_token", frac=0.25)
    done.append(d); jt.append(j)
    print(f"  {lab:12s} {d:5.1f} {j:.1f}")
fig, ax = plt.subplots(1, 2, figsize=(COLW, 2.0))
labels = ["no\ncache", "GPU\ncache", "160\nGB", "300\nGB"]
cols = [COL[p[2]] for p in pol]
for a, vals, ylab, top, fmt in ((ax[0], done, "requests finished (%)", 112, "{:.0f}"),
                                (ax[1], jt, "GPU energy per prompt token (mJ)", 118, "{:.1f}")):
    a.bar(range(4), vals, width=0.66, color=cols, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        a.text(i, v + top * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK)
    a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=7, linespacing=1.0); a.tick_params(axis="x", length=0)
    a.set_ylabel(ylab); a.set_ylim(0, top); a.grid(False)
fig.tight_layout(pad=0.3, w_pad=1.0); fig.savefig(OUT + "production.pdf"); plt.close(fig)

# ---------- Figure: adaptive gate against the GPU cache alone (Table 6, 150 s steps), mean of the two rounds
ad8 = load("phase8-results.zip", "agents-8b-adaptive.json")
groups = [("Qwen3-8B", b8, "vllm-gpu-8b", ad8, "vllm-adaptive-8b", [64, 128, 256]),
          ("Qwen3-32B, 27k", lc, "vllm-gpu-lc", lc, "vllm-adaptive-lc", [4, 8, 16, 32])]
fig, ax = plt.subplots(1, 3, figsize=(TEXTW, 2.3), gridspec_kw={"width_ratios": [3, 4, 5.2]})
w = 0.36
print("adaptive: sessions, GPU J (min-max), adaptive J (min-max), GPU p90, adaptive p90")
rows = []
for k, (title, Rg, g, Ra, ad, xs) in enumerate(groups):
    a = ax[k]
    for i, n in enumerate(xs):
        gv, av = rounds(Rg, g, "j_per_turn", n), rounds(Ra, ad, "j_per_turn", n)
        a.bar(i - w / 2 - 0.01, statistics.mean(gv), w, color=COL["gpu"], label="GPU cache" if i == 0 else None)
        a.bar(i + w / 2 + 0.01, statistics.mean(av), w, color=COL["gate"], label="adaptive gate" if i == 0 else None)
        gp, ap = rounds(Rg, g, "resume_ttft_p90", n), rounds(Ra, ad, "resume_latency_p90", n)
        rows.append((title, n, statistics.mean(gp), statistics.mean(ap)))
        print(f"  {title[:9]:9s} {n:4d} {statistics.mean(gv):7.1f} ({min(gv):.1f}-{max(gv):.1f}) "
              f"{statistics.mean(av):7.1f} ({min(av):.1f}-{max(av):.1f}) {statistics.mean(gp):7.2f} {statistics.mean(ap):6.2f}")
    a.set_xticks(range(len(xs))); a.set_xticklabels([str(n) for n in xs]); a.tick_params(axis="x", length=0)
    a.set_title(title, fontsize=8, pad=4); a.set_xlabel("sessions"); a.set_ylim(0, None)
ax[0].set_ylabel("GPU energy per turn (J)"); ax[1].set_ylabel("GPU energy per turn (J)")
# right panel: dumbbell of 90th-percentile resume latency, one row per cell, log scale
a = ax[2]
a.grid(False); a.grid(True, axis="x", color="#ecebe8", lw=0.6)
yy = list(range(len(rows)))[::-1]
for y, (title, n, gp, ap) in zip(yy, rows):
    a.plot([gp, ap], [y, y], color=RULE, lw=1.2, zorder=1)
    a.plot(gp, y, "o", ms=4.5, color=COL["gpu"], zorder=2, mec="white", mew=0.6)
    a.plot(ap, y, "o", ms=4.5, color=COL["gate"], zorder=2, mec="white", mew=0.6)
    fmt = (lambda v: f"{v:.2g}" if v < 10 else f"{v:.0f}")
    lo, hi = min(gp, ap), max(gp, ap)
    if hi / lo < 1.08:      # the two policies coincide: one label
        a.text(hi * 1.22, y, f"{fmt(hi)} (both)", ha="left", va="center", fontsize=6.8, color=INK)
    else:
        a.text(lo / 1.22, y, fmt(lo), ha="right", va="center", fontsize=6.8, color=INK)
        a.text(hi * 1.22, y, fmt(hi), ha="left", va="center", fontsize=6.8, color=INK)
a.set_yticks(yy); a.set_yticklabels([f"{'8B' if t.startswith('Qwen3-8B') else '32B, 27k'}, $N$ = {n}" for t, n, _, _ in rows])
a.tick_params(axis="y", length=0)
a.axhline(3.5, color=RULE, lw=0.6)
a.set_xscale("log"); a.set_xlim(0.18, 600)
a.xaxis.set_major_locator(mticker.FixedLocator([1, 10, 100]))
a.xaxis.set_major_formatter(mticker.FixedFormatter(["1", "10", "100"]))
a.xaxis.set_minor_locator(mticker.NullLocator())
a.set_xlabel("90th-percentile resume latency (s, log scale)")
a.set_title("resume latency", fontsize=8, pad=4)
a.spines["left"].set_visible(False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=0.3, rect=[0, 0.07, 1, 1], w_pad=1.6); fig.savefig(OUT + "adaptive.pdf"); plt.close(fig)
print("ok")
