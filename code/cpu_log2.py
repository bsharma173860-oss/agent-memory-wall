import os, sys, time
TCK = os.sysconf("SC_CLK_TCK"); SKIP = ("cpu_log", "agent_energy", "engine_energy")
def engine_cpu_s():
    tot = 0.0
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore").lower()
            if ("vllm" in cmd or "sglang" in cmd) and not any(s in cmd for s in SKIP):
                f = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
                tot += (int(f[11]) + int(f[12])) / TCK
        except (OSError, IndexError, ValueError): pass
    return tot
def host():
    v = list(map(int, open("/proc/stat").readline().split()[1:8])); return sum(v), v[3] + v[4]
out = open(sys.argv[1], "a"); out.write("unix_s,engine_cpu_s,host_busy_frac,agents_json_mtime\n")
t0, i0 = host()
while True:
    time.sleep(1); t1, i1 = host()
    busy = 1 - (i1 - i0) / max(t1 - t0, 1); t0, i0 = t1, i1
    try: m = os.path.getmtime("/workspace/agents.json")
    except OSError: m = ""
    out.write(f"{time.time():.3f},{engine_cpu_s():.2f},{busy:.4f},{m}\n"); out.flush()
