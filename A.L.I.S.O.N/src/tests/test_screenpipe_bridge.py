"""W2 verification: ScreenpipeBridge against a mocked Screenpipe REST endpoint.

Covers: transition-gated writes, dedupe, capture_policy exclusion, and silent
degradation when Screenpipe is offline. Uses a fake memory index so the real
HippocampalMemoryIndex / alison_core is never imported (no brain side effects).
"""
import os
import sys
import json
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from http.server import BaseHTTPRequestHandler, HTTPServer

import alison_screenpipe as sp

CURRENT = {}
PORT = 8099


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = CURRENT.get("payload")
        body = json.dumps(payload).encode() if payload else b'{"data":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FakeMemory:
    def __init__(self):
        self.writes = 0
        self.texts = []

    def write_fast_weight(self, key, val):
        self.writes += 1

    def store(self, embedding, text, valence=0.0):
        self.texts.append(text)

    def write_episodic(self, key, val, index):
        pass


def frame(app, window, text, url=""):
    return {"data": [{"content": {
        "app_name": app, "window_name": window, "text": text, "url": url}}]}


def main():
    cap_path = os.path.join(os.path.dirname(__file__), "..", "config", "capture_policy.json")
    server = HTTPServer(("127.0.0.1", PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    mem = FakeMemory()
    bridge = sp.ScreenpipeBridge(mem, url=f"http://127.0.0.1:{PORT}",
                                 capture_policy_path=cap_path)

    ok = True

    CURRENT["payload"] = frame("VSCode", "main.py", "def foo(): pass")
    if not bridge.process_and_ingest_step():
        print("FAIL: first frame not ingested"); ok = False
    if bridge.process_and_ingest_step():
        print("FAIL: duplicate frame ingested (no dedupe)"); ok = False
    if mem.writes != 1:
        print(f"FAIL: expected 1 write, got {mem.writes}"); ok = False

    CURRENT["payload"] = frame("VSCode", "main.py", "def bar(): pass")
    if not bridge.process_and_ingest_step():
        print("FAIL: changed text not ingested"); ok = False
    if mem.writes != 2:
        print(f"FAIL: expected 2 writes, got {mem.writes}"); ok = False

    CURRENT["payload"] = frame("KeePass.exe", "vault", "password secret")
    if bridge.process_and_ingest_step():
        print("FAIL: excluded app (KeePass.exe) was ingested"); ok = False

    CURRENT["payload"] = frame("Chrome", "Bank Account", "balance 1000")
    if bridge.process_and_ingest_step():
        print("FAIL: excluded window (Bank Account) ingested"); ok = False

    server.shutdown()
    server.server_close()
    if bridge.process_and_ingest_step():
        print("FAIL: offline Screenpipe did not degrade silently"); ok = False

    if ok:
        print("SCREENPIPE BRIDGE OK: transition-gated writes=%d, dedupe+exclusion+offline all handled"
              % mem.writes)
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
