"""
SCREEN SENSE -- Visual Context Pipeline (Phases 33 + 34)
========================================================
- Lazy UIA active-window text extraction (pywinauto, optional)
- Lazy DeBERTa zero-shot activity classifier (transformers, CPU)
- Lazy Go-Emotions screen->limbic-state router (transformers, CPU)
- Background daemon refreshing `current_context` every 10s

All heavy dependencies load lazily on first use, so the ICA can boot
without pywinauto or cached HF models.
"""

import threading
import os
import torch

DEBUG = os.environ.get("SCREEN_SENSE_DEBUG") == "1"

def _dbg(msg):
    if DEBUG:
        print(f"  [SCREEN_SENSE] {msg}", flush=True)

current_context = "Visual context unavailable: No active window."
_window_name = "unknown"
_stop = threading.Event()
_activity_classifier = None
_emotion_classifier = None
_activity_lock = threading.Lock()
_emotion_lock = threading.Lock()

_ACTIVITIES = [
    "writing code",
    "writing a document",
    "web browsing",
    "watching a video stream",
    "playing a video game",
    "reading email",
    "system error message",
    "idle desktop",
]

DEVICE = torch.device("cpu")


def _extract_window_text():
    """UIA-based extraction of the active window's title + visible text."""
    global _window_name
    _dbg("UIA extraction start")
    try:
        from pywinauto import Desktop

        desktop = Desktop(backend="uia")
        try:
            active = desktop.windows(active_only=True)
        except Exception:
            active = None
        if not active:
            active = [w for w in desktop.windows() if w.is_visible()]
        if not active:
            return ""
        w = active[0]
        _window_name = (w.window_text() or "unknown")[:60]
        parts = [_window_name]
        try:
            elems = w.descendants(control_type="Text", depth=6, timeout=2)
        except Exception:
            elems = []
        for e in elems[:15]:
            try:
                t = e.window_text()
            except Exception:
                t = ""
            if t and t.strip() and t not in parts:
                parts.append(t.strip())
        return " | ".join(parts)[:2000]
    except Exception:
        return ""
    finally:
        _dbg("UIA extraction done")


def _classify_activity(raw_text):
    """DeBERTa zero-shot activity label (lazy, CPU)."""
    global _activity_classifier
    if _activity_classifier is None:
        with _activity_lock:
            if _activity_classifier is None:
                from transformers import pipeline

                _activity_classifier = pipeline(
                    "zero-shot-classification",
                    model="cross-encoder/nli-deberta-v3-xsmall",
                    device=-1,
                )
    res = _activity_classifier(raw_text[:500], _ACTIVITIES, multi_label=False)
    if isinstance(res, list) and res and isinstance(res[0], dict):
        res = res[0]
    _dbg(f"activity={res['labels'][0]}")
    return res["labels"][0]


def get_visual_context():
    """Extract active-window text via UIA and classify its activity."""
    raw_text = _extract_window_text()
    if not raw_text.strip():
        return "Visual context unavailable: No active window."
    try:
        activity = _classify_activity(raw_text)
    except Exception:
        activity = "unknown"
    preview = raw_text[:500]
    return f"Window: {_window_name}\nActivity: {activity}\nText: {preview}"


# ==================================================================
# PHASE 34.2: GO-EMOTIONS -> 6-DIM LIMBIC-STATE ROUTING
# Hunger (dim 0) is physiological (gridworld battery) -- never set here.
# ==================================================================
_EMOTION_DIMS = {
    "pain": {"dim": 1, "labels": {"anger": 0.9, "annoyance": 0.9, "grief": 0.9}},
    "fatigue": {"dim": 2, "labels": {"boredom": 0.9, "neutral": 0.5}},
    "curious": {"dim": 3, "labels": {"curiosity": 1.0, "excitement": 0.8, "amusement": 0.5}},
    "anxious": {
        "dim": 4,
        "labels": {"fear": 0.8, "nervousness": 0.8, "confusion": 0.5, "embarrassment": 0.5},
    },
    "altruistic": {
        "dim": 5,
        "labels": {"love": 0.9, "gratitude": 0.9, "caring": 0.8, "admiration": 0.7},
    },
}


def _get_emotion_scores(text):
    """All-28-label Go-Emotions scores, defensively normalized across
    transformers version differences in single-input pipeline shape."""
    global _emotion_classifier
    if _emotion_classifier is None:
        with _emotion_lock:
            if _emotion_classifier is None:
                from transformers import pipeline

                _emotion_classifier = pipeline(
                    "text-classification",
                    model="SamLowe/roberta-base-go_emotions",
                    top_k=None,
                    device=-1,
                )
    out = _emotion_classifier(text[:500])
    if out and isinstance(out[0], list):
        out = out[0]
    _dbg("emotion scores computed")
    return {r["label"]: r["score"] for r in out}


def get_pc_state_from_context(context):
    """Screen context -> 6-dim cognitive PC state (clamped [0,1]).

    Maps Go-Emotions label scores onto the cognitive dimensions of the
    limbic vector. Hunger (dim 0) is left untouched: it is a physiological
    drive fed by the gridworld battery loop, not by screen content.
    """
    v = torch.zeros(6, dtype=torch.float32)
    if not context or "No active window" in context:
        return v
    text = context.split("Text: ")[-1].strip()
    if not text:
        return v
    try:
        emotion_scores = _get_emotion_scores(text)
    except Exception:
        return v
    for mapping in _EMOTION_DIMS.values():
        val = sum(mult * emotion_scores.get(label, 0.0)
                  for label, mult in mapping["labels"].items())
        v[mapping["dim"]] = max(0.0, min(1.0, val))
    return v


def screen_daemon():
    """Refresh the global screen context every 10 seconds."""
    global current_context
    _dbg("daemon started")
    while not _stop.wait(10.0):
        _dbg("refresh cycle begin")
        try:
            ctx = get_visual_context()
            _dbg("context fetched")
            if ctx and ctx != current_context:
                current_context = ctx
                _dbg("context updated")
        except Exception as e:
            _dbg(f"context fetch failed: {e}")


if __name__ == "__main__":
    ctx = get_visual_context()
    print(ctx)
    print("PC_STATE:", get_pc_state_from_context(ctx).tolist())
