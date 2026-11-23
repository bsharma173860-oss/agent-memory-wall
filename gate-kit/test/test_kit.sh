#!/bin/bash
# The kit's own test, on a machine with no GPU (about 3 minutes):
#  1. the summarizer, run on the paper's own 10-minute results (results/phase9-results.zip), gives the paper's numbers;
#  2. the session sizes for the H200 are the paper's;
#  3. the whole kit runs end to end on the stand-in engine (code/fake, vLLM and SGLang) with a stand-in meter;
#  4. the AMD stand-in for nvidia-smi answers from a stand-in rocm-smi.
# Usage: bash gate-kit/test/test_kit.sh   (python3 with the standard library only)
set -euo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd); ROOT=$(dirname "$KIT"); PY=${PY:-python3}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

echo "1. summarizer against the paper's 10-minute runs"
"$PY" -I -m zipfile -e "$ROOT/results/phase9-results.zip" "$T/p9"
for f in agents-p9-8b agents-p9-lc; do "$PY" -I "$KIT/summarize.py" "$T/p9/$f.json" --out "$T/$f.kit.json" > /dev/null; done
"$PY" -I - "$T" <<'PY'
import json, sys
T = sys.argv[1]
# the paper's Table (600 s steps) and its text: turns, J/turn, served, first-turn p90 (s), resumed-turn p90 (s)
want = {"agents-p9-8b": [(1436, 292, 256, 68, 102), (2119, 168, 256, 183, 226)],
        "agents-p9-lc": [(143, 3022, 32, 131, 130), (232, 1507, 32, 253, 251)]}
for f, rows in want.items():
    S = json.load(open(f"{T}/{f}.kit.json"))
    got = [(p["turns"], round(p["gpu_j_per_turn"]), p["sessions_served"], round(p["first_turn_latency_p90_s"]), round(p["resume_latency_p90_s"]))
           for p in S["policies"]]
    assert got == rows, (f, got, rows)
    g = S["gate_vs_baseline"]
    print(f"   {f}: {got} ok; turns {g['turns_change_pct']:+}%, J/turn {g['gpu_j_per_turn_change_pct']:+}%")
PY

echo "2. sizes"
"$PY" -I - "$KIT" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import sizes
assert sizes.size("8b", sizes.GPUS["h200"])["sessions"] == 256 and sizes.size("32b-long", sizes.GPUS["h200"])["sessions"] == 32
assert sizes.size("32b-long", sizes.GPUS["h100"])["sessions"] is None
print("   H200 gives the paper's 256 and 32; 32b-long does not fit one H100: ok")
PY

echo "3. end to end on the stand-in engine"
for E in vllm sglang; do
  PYTHONPATH="$KIT/test/fakejoules:$ROOT/code/fake" PATH="$KIT/test/fakebin:$PATH" "$PY" "$KIT/run_kit.py" --engine $E \
    --engine-python "$PY" --workload 8b --sessions 12 --duration 20 --capacity 20000 --pauses 1 --tool-tokens 100 \
    --answer-tokens 20 --idle 1 --cooldown 1 --settle-window 1 --startup-timeout 120 --port 187$((RANDOM % 90 + 10)) \
    --out-dir "$T/run-$E" > "$T/run-$E.log" 2>&1 || { tail -40 "$T/run-$E.log"; exit 1; }
  "$PY" -I - "$T/run-$E/gate-kit-results.json" "$E" <<'PY'
import json, sys
S = json.load(open(sys.argv[1])); E = sys.argv[2]
assert not S["errors"] and not S["skipped"], (S["errors"], S["skipped"])
assert [p["policy"] for p in S["policies"]] == ["gpu-only", "brain-v0-adaptive"], S["policies"]
for p in S["policies"]:
    assert p["turns"] > 0 and p["failed_requests"] == 0 and p["sessions_served"] > 0, p
    for k in ("cache_hit", "first_turn_latency_p50_s", "first_turn_latency_p90_s", "resume_latency_p50_s", "resume_latency_p90_s", "gpu_j_per_turn"):
        assert p[k] is not None, (k, p)
gate = S["policies"][1]["gate"]
assert gate["kv_capacity_tokens"] == 20000 and gate["theta"] == 1.2 and gate["share_of_intervals_gate_on"] > 0, gate
assert S["gate_vs_baseline"]["turns_change_pct"] is not None and S["meter"]["gpu_energy"] == "measured"
assert S["kit_run"]["sessions"] == 12 and S["kit"] == "gate-kit/1"
import os
assert not any(os.path.exists("/" + f) for f in ("gate-kit-raw.json", "gate-kit-raw.csv", "gate-kit-raw-timeline.json")), "wrote into /"
print(f"   {E}: both policies ran, gate on in {gate['share_of_intervals_gate_on']:.0%} of intervals, results file complete: ok")
PY
done

echo "4. the AMD stand-in for nvidia-smi"
mkdir -p "$T/rocm"; cat > "$T/rocm/rocm-smi" <<'SH'
#!/bin/sh
case "$*" in
  *showmeminfo*) echo '{"card0": {"VRAM Total Memory (B)": "206141652992", "VRAM Total Used Memory (B)": "1073741824"}}';;
  *showtemp*) echo '{"card0": {"Temperature (Sensor edge) (C)": "40.0", "Temperature (Sensor junction) (C)": "45.0"}}';;
  *) echo '{"card0": {"Card Series": "AMD Instinct MI300X"}}';;
esac
SH
chmod +x "$T/rocm/rocm-smi"
used=$(PATH="$T/rocm:$PATH" "$KIT/amd/nvidia-smi" -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
temp=$(PATH="$T/rocm:$PATH" "$KIT/amd/nvidia-smi" -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits)
[ "$used" = "1024" ] && [ "$temp" = "45.0" ] || { echo "AMD stand-in: got '$used' '$temp'"; exit 1; }
echo "   memory.used 1024 MiB and temperature 45.0 C from rocm-smi: ok"
echo "ALL KIT TESTS PASSED"
