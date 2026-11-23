#!/bin/bash
# Download the production trace used in the paper (Mooncake / Kimi tool-agent trace, released by Moonshot AI under
# the Mooncake repository's license) and check that it is the same file.
set -euo pipefail
mkdir -p traces
curl -sSL -o traces/toolagent_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl
echo "48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71  traces/toolagent_trace.jsonl" | sha256sum -c -
