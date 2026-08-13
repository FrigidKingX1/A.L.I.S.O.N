"""alison_screenpipe.py -- optional Screenpipe ingestion adapter for A.L.I.S.O.N.

Routes Screenpipe's local REST OCR feed (http://localhost:3030) into the
real HippocampalMemoryIndex. Flag-gated and OFF by default: alison_sense.py
remains the offline perception default. Degrades silently when Screenpipe is
not running. Exclusion filtering is sourced from config/capture_policy.json
(W0) so sensitive apps/windows/urls are never written to memory.
"""
import os
import json
import hashlib
import threading
import urllib.request
import urllib.error

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SP_URL = "http://localhost:3030"


class ScreenpipeBridge:
    def __init__(self, memory_index, url=DEFAULT_SP_URL, device="cpu",
                 capture_policy_path=None):
        self.memory_index = memory_index
        self.url = url
        self.device = device
        self.key_projector = nn.Linear(256, 128, bias=False).to(device)
        if capture_policy_path is None:
            capture_policy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config", "capture_policy.json")
        self.exclusions = self._load_capture_policy(capture_policy_path)
        self._last_hash = ""
        self._last_app = ""

    def _load_capture_policy(self, path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data.get("exclusions", {})
        except Exception:
            return {}

    def is_excluded(self, app="", window="", url=""):
        def _match(lst, val):
            if not val:
                return False
            val = val.lower()
            return any(s and s.lower() in val for s in (lst or []))
        return (_match(self.exclusions.get("apps"), app)
                or _match(self.exclusions.get("windows"), window)
                or _match(self.exclusions.get("urls"), url))

    def _fetch_latest_frame(self):
        try:
            req = urllib.request.Request(
                f"{self.url}/search?limit=1&type=ocr",
                headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and data.get("data"):
                return data["data"][0]
        except Exception:
            return None
        return None

    def _extract(self, frame):
        content = frame.get("content", frame) if isinstance(frame, dict) else {}
        if not isinstance(content, dict):
            content = {}
        app = content.get("app_name", frame.get("app_name", ""))
        window = content.get("window_name", frame.get("window_name", ""))
        text = content.get("text", frame.get("text", ""))
        url = content.get("url", frame.get("url", ""))
        return app or "", window or "", text or "", url or ""

    def _generate_key_embedding(self, app, window, text):
        combined = f"[{app}]::{window}::{text[:100]}"
        hist = torch.zeros(256, dtype=torch.float32)
        for ch in combined.encode("utf-8", "ignore"):
            hist[ch % 256] += 1.0
        hist = F.normalize(hist.unsqueeze(0), p=2, dim=-1).to(self.device)
        with torch.no_grad():
            key = F.normalize(self.key_projector(hist), p=2, dim=-1)
        return key.squeeze(0)

    def process_and_ingest_step(self):
        frame = self._fetch_latest_frame()
        if not frame:
            return False
        app, window, text, url = self._extract(frame)
        if not text.strip():
            return False
        if self.is_excluded(app, window, url):
            return False
        curr_hash = hashlib.md5(
            f"{app}:{window}:{text[:200]}".encode("utf-8")).hexdigest()
        if curr_hash == self._last_hash:
            return False
        is_app_switch = app != self._last_app
        self._last_hash = curr_hash
        self._last_app = app
        key = self._generate_key_embedding(app, window, text)
        val = self._generate_key_embedding("VALUE", app, text[::2])
        self.memory_index.write_fast_weight(key, val)
        self.memory_index.store(key, f"[{app}] {window}: {text[:200]}", valence=0.0)
        if is_app_switch and hasattr(self.memory_index, "write_episodic"):
            try:
                idx = len(getattr(self.memory_index, "texts", [])) - 1
                self.memory_index.write_episodic(key, val, max(0, idx))
            except Exception:
                pass
        return True

    def daemon(self, stop_ev, poll_interval=5.0, is_enabled=None):
        while not stop_ev.is_set():
            try:
                if is_enabled is None or is_enabled():
                    self.process_and_ingest_step()
            except Exception:
                pass
            stop_ev.wait(poll_interval)
