"""alison_actions.py -- Safe OS Action Executor for A.L.I.S.O.N.

Lets the engine drive Windows UI Automation (pywinauto / comtypes / UIA) for a
small, explicitly allow-listed set of operations. Every action must be emitted
by the LLM as a GBNF-constrained JSON object (see ``GBNF_SCHEMA``); arbitrary
shell strings are never executed. Dependencies are imported lazily so the
module imports cleanly without pywinauto installed.

Security model:
  * Only the actions in ``ALLOWED_ACTIONS`` are permitted.
  * ``type_text`` only injects into the *currently focused* window.
  * ``open_app`` is restricted to an ``APP_ALLOWLIST`` (extend as needed).
"""

import json
import os

# GBNF grammar for llama.cpp constrained decoding. Produces JSON of the form:
#   {"action": "focus_window", "args": {"title": "Notepad"}}
GBNF_SCHEMA = r"""
root ::= object
object ::= "{" ws keyvalue ws "}"
keyvalue ::= "\"action\"" ws ":" ws action ws "," ws "\"args\"" ws ":" ws args
action ::= "\"open_app\"" | "\"focus_window\"" | "\"type_text\"" | "\"click\""
args ::= "{" ws argpair ws "}"
argpair ::= "\"title\"" ws ":" ws string
          | "\"name\"" ws ":" ws string
          | "\"text\"" ws ":" ws string
string ::= "\"" [^"]* "\""
ws ::= [ \t\n]*
"""

ALLOWED_ACTIONS = {"open_app", "focus_window", "type_text", "click"}

# Applications the engine is permitted to launch. Add entries as the product
# matures; anything not listed is rejected by execute().
APP_ALLOWLIST = {
    "notepad", "notepad.exe", "calc", "calc.exe", "mspaint", "mspaint.exe",
    "chrome", "chrome.exe", "firefox", "firefox.exe", "edge", "msedge.exe",
    "explorer", "explorer.exe", "alison_gui", "alison_gui.exe",
}

# Paths searched when launching an allow-listed app name.
APP_DIRS = [
    os.environ.get("SystemRoot", r"C:\Windows"),
    os.environ.get("ProgramFiles", r"C:\Program Files"),
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
]


class ActionExecutor:
    def __init__(self):
        self._app = None

    @property
    def available(self):
        try:
            import pywinauto  # noqa: F401
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    @staticmethod
    def parse_action(llm_text):
        """Extract the first JSON action object from LLM output, or None."""
        start = llm_text.find("{")
        end = llm_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(llm_text[start:end + 1])
        except Exception:
            return None
        if not isinstance(obj, dict) or "action" not in obj:
            return None
        return obj

    def _resolve_app_path(self, name):
        base = name if name.lower().endswith(".exe") else name + ".exe"
        for d in APP_DIRS:
            if not d:
                continue
            cand = os.path.join(d, base)
            if os.path.exists(cand):
                return cand
        # Fall back to PATH lookup via createfile
        return name

    def execute(self, action_obj):
        """Validate and perform a single action dict. Returns a status string."""
        if not isinstance(action_obj, dict):
            return "rejected: not an object"
        action = action_obj.get("action")
        if action not in ALLOWED_ACTIONS:
            return f"rejected: unknown action '{action}'"
        args = action_obj.get("args", {}) or {}

        if action == "open_app":
            name = args.get("name", "")
            if name.lower() not in APP_ALLOWLIST:
                return f"rejected: app '{name}' not allow-listed"
            import pywinauto
            path = self._resolve_app_path(name)
            pywinauto.Application().start(path)
            return f"opened: {name}"

        if action == "focus_window":
            title = args.get("title", "")
            import pywinauto
            app = pywinauto.Application().connect(title_re=title, timeout=3)
            app.top_window().set_focus()
            return f"focused: {title}"

        if action == "type_text":
            text = args.get("text", "")
            import pywinauto
            # Only type into the currently focused control (no target spoofing).
            pywinauto.keyboard.send_keys(text)
            return f"typed {len(text)} chars into focused window"

        if action == "click":
            import pywinauto
            pywinauto.mouse.click()
            return "clicked at cursor"

        return "rejected: unhandled"

    def execute_llm(self, llm_text):
        obj = self.parse_action(llm_text)
        if obj is None:
            return "no action found"
        return self.execute(obj)
