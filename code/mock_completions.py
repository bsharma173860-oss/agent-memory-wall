"""Tiny OpenAI-compatible mock (/v1/models, /v1/completions, /health) for wiring tests of the BFCL harness."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send({"data": [{"id": "Qwen/Qwen3-32B"}, {"id": "bench"}]})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        self._send({"id": "x", "object": "text_completion", "model": "Qwen/Qwen3-32B",
                    "choices": [{"index": 0, "text": "I cannot help with that.", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16}})
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
