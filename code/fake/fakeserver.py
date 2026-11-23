"""A fake OpenAI-compatible engine for testing the benchmark scripts without a GPU.

Streams N tokens, reports usage with cached prompt tokens, exposes /metrics counters, and keeps an LRU prefix cache
with a token capacity: --no-enable-prefix-caching / --disable-radix-cache turn the cache off, and the host-offload
flags (--kv-offloading-size, --enable-hierarchical-cache) add a slower second tier four times the size."""
import json, os, sys, threading, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
FLAGS = ["--port", "--model", "--model-path", "--max-num-batched-tokens", "--max-num-seqs", "--chunked-prefill-size",
         "--max-running-requests", "--schedule-policy", "--no-enable-prefix-caching", "--disable-radix-cache",
         "--kv-offloading-size", "--kv-offloading-backend", "--enable-hierarchical-cache", "--hicache-size", "--hicache-ratio"]
if "--help" in args:
    print("options: " + " ".join(FLAGS)); sys.exit(0)
port = int(args[args.index("--port") + 1])
NOCACHE = "--no-enable-prefix-caching" in args or "--disable-radix-cache" in args
OFFLOAD = "--kv-offloading-size" in args or "--enable-hierarchical-cache" in args
CAP = 40000                                      # tokens of prefix cache on the "GPU"
HOST = 4 * CAP if OFFLOAD else 0                 # tokens in the host tier
PREFILL_S, RESTORE_S, DECODE_S = 0.00002, 0.000004, 0.002   # seconds per uncached token, restored token, output token
lock = threading.Lock()
gpu, host = OrderedDict(), OrderedDict()         # prompt -> tokens, least recently used first
C = {"prompt_tokens_total": 0.0, "generation_tokens_total": 0.0, "prefix_cache_queries_total": 0.0,
     "prefix_cache_hits_total": 0.0, "num_requests_running": 0.0}


def lookup(p, pt, per=3.2):
    """(cached tokens from the GPU tier, restored tokens from the host tier); then store p in the GPU tier."""
    best_g = max([len(os.path.commonprefix([p, q])) for q in gpu] or [0])
    best_h = max([len(os.path.commonprefix([p, q])) for q in host] or [0])
    g = int(best_g / per); h = max(0, int(best_h / per) - g)
    gpu[p] = pt; gpu.move_to_end(p)
    while sum(gpu.values()) > CAP and len(gpu) > 1:
        q, n = gpu.popitem(last=False)
        if HOST:
            host[q] = n
            while sum(host.values()) > HOST and host:
                host.popitem(last=False)
    return g, h


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/v1/models":
            b = b'{"data":[{"id":"bench"}]}'
        elif self.path == "/metrics":
            b = "".join(f"vllm:{k} {v}\n" for k, v in C.items()).encode()
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = d["max_tokens"]
        if isinstance(d.get("prompt"), list):    # /v1/completions with token ids: one character per token
            p = "".join(chr(256 + t % 60000) for t in d["prompt"]); pt, per = len(d["prompt"]), 1.0
        else:
            p = "".join(m["content"] for m in d["messages"]); pt, per = int(len(p) / 3.2), 3.2
        text_mode = "prompt" in d
        with lock:
            g, h = (0, 0) if NOCACHE else lookup(p, pt, per)
            C["prompt_tokens_total"] += pt; C["prefix_cache_queries_total"] += pt; C["prefix_cache_hits_total"] += g + h
            C["num_requests_running"] += 1
        time.sleep((pt - g - h) * PREFILL_S + h * RESTORE_S)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def send(obj):
            s = f"data: {json.dumps(obj) if obj != '[DONE]' else obj}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(s), s))
        for _ in range(n):
            time.sleep(DECODE_S); send({"choices": [{"text": "x"}]} if text_mode else {"choices": [{"delta": {"content": "x"}}]})
        send({"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": n, "prompt_tokens_details": {"cached_tokens": g + h}}})
        send("[DONE]"); self.wfile.write(b"0\r\n\r\n")
        with lock:
            C["generation_tokens_total"] += n; C["num_requests_running"] -= 1


time.sleep(1)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
