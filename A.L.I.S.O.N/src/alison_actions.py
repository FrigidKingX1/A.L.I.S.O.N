"""alison_actions.py -- Safe OS Action Executor for A.L.I.S.O.N. (v3).

Universal, app-agnostic action layer. Every action is emitted by the LLM as a
GBNF-constrained JSON object and validated here before anything touches the OS.
All heavy dependencies (pywinauto, keyboard, win32gui) are imported lazily so
the module imports cleanly without them installed.

Security model (W3 / §5.3):
  * Three tiers: passive (read-only), reversible (safe OS mutations), privileged
    (shell / destructive). Privileged actions require either an allowlist regex
    match OR explicit confirmation; otherwise they are denied by default.
  * `execute_cmd` runs sandboxed: restricted working directory, optional
    dry-run preview for filesystem-mutating commands, and a tracked subprocess
    handle so the kill switch can terminate an in-flight command.
  * A global kill switch (Core IPC `kill_switch` / GUI Ctrl+Alt+K) sets
    `action_executor_enabled = False` and calls `terminate_active()`.
"""
import json
import os
import re
import shlex
import subprocess
import tempfile

try:
    import win32job
    import win32con
    HAS_WIN32_JOB = True
except ImportError:
    HAS_WIN32_JOB = False

GBNF_SCHEMA = r"""
root ::= object
object ::= "{" ws keyvalue ws "}"
keyvalue ::= "\"action\"" ws ":" ws action ws "," ws "\"args\"" ws ":" ws args
action ::= "\"open_app\"" | "\"launch_app\"" | "\"focus_window\"" | "\"type_text\"" | "\"input_key\"" | "\"read_screen\"" | "\"read_file\"" | "\"write_file\"" | "\"click\"" | "\"delete_file\"" | "\"execute_cmd\"" ws "," ws "\"args\"" ws ":" ws args
args ::= "{" ws argpair ws "}"
argpair ::= "\"title\"" ws ":" ws string
          | "\"name\"" ws ":" ws string
          | "\"text\"" ws ":" ws string
          | "\"sequence\"" ws ":" ws string
          | "\"path\"" ws ":" ws string
          | "\"content\"" ws ":" ws string
          | "\"command\"" ws ":" ws string
string ::= "\"" [^"]* "\""
ws ::= [ \t\n]*
"""

ALLOWED_ACTIONS = {
    "open_app", "launch_app", "focus_window", "type_text", "input_key",
    "read_screen", "read_file", "write_file", "click", "delete_file",
    "execute_cmd",
}

APP_ALLOWLIST = {
    "notepad", "notepad.exe", "calc", "calc.exe", "mspaint", "mspaint.exe",
    "chrome", "chrome.exe", "firefox", "firefox.exe", "edge", "msedge.exe",
    "explorer", "explorer.exe", "alison_gui", "alison_gui.exe",
}

APP_DIRS = [
    os.environ.get("SystemRoot", r"C:\Windows"),
    os.environ.get("ProgramFiles", r"C:\Program Files"),
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
]


class ActionExecutor:
    def __init__(self, policy_path=None):
        self._app = None
        self._active_process = None
        self.enabled = True
        if policy_path is None:
            policy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config", "action_policy.json")
        self.policy = self._load_policy(policy_path)
        self.global_dry_run = os.environ.get("ALISON_GLOBAL_DRY_RUN", "").lower() in (
            "1", "true", "yes")
        self._job_handle = None
        self._setup_job_object()
        signing = self.policy.get("signing", {}) or {}
        self._signing_enabled = bool(signing.get("enabled", False))
        self._signing_key = os.environ.get("ALISON_POLICY_KEY", "")
        self._signature_valid = None

    def _setup_job_object(self):
        """Win32 Job Object with KILL_ON_JOB_CLOSE (Phase 2 hardening).

        Every execute_cmd child is assigned to the job; when the Core process
        dies (crash or exit), the kernel closes the job handle and kills the
        entire action process tree -- no orphans survive the engine. The kill
        switch uses the same handle to terminate the whole tree at once.
        """
        if not HAS_WIN32_JOB:
            return
        try:
            create_fn = getattr(win32job, "CreateJobObjectW", None) or win32job.CreateJobObject
            job = create_fn(None, "ALISON_ACTION_JOB")
            info = win32job.QueryInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation)
            info["BasicLimitInformation"]["LimitFlags"] |= (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
            win32job.SetInformationJobObject(
                job, win32job.JobObjectExtendedLimitInformation, info)
            self._job_handle = job
            print("[ACTIONS] Win32 Job Object active (KILL_ON_JOB_CLOSE) -- "
                  "action process trees are contained.")
        except Exception as exc:
            self._job_handle = None
            print(f"[ACTIONS][warn] job object unavailable, falling back to "
                  f"direct process kill: {exc}")

    def _assign_to_job(self, proc):
        if not self._job_handle or proc is None or proc.poll() is not None:
            return
        try:
            win32job.AssignProcessToJobObject(self._job_handle, int(proc._handle))
        except Exception:
            pass

    def _resume_suspended(self, proc):
        try:
            import ctypes
            ctypes.windll.ntdll.NtResumeProcess(int(proc._handle))
        except Exception:
            try:
                proc.resume()  # psutil-style fallback, if available
            except Exception:
                pass

    def _kill_job_tree(self):
        if self._job_handle:
            try:
                win32job.TerminateJobObject(self._job_handle, 1)
                return True
            except Exception:
                pass
        return False

    def _verify_policy_signing(self):
        """Tier 3 signing scaffold: when policy signing is enabled, privileged
        actions require a valid HMAC-SHA256 policy signature. Deny-by-default
        when no key/signature is present."""
        if not self._signing_enabled:
            return True
        if self._signature_valid is None:
            try:
                from alison_signing import verify_policy_signature
                self._signature_valid = verify_policy_signature(
                    self.policy, self._signing_key)
            except Exception:
                self._signature_valid = False
        return self._signature_valid

    def _load_policy(self, path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    # -- capability tier selection (universal 3-tier) ----------------------
    def select_capability_tier(self, app=None, action=None):
        integrations = self.policy.get("app_integrations", {})
        if app and app.lower() in {k.lower() for k in integrations}:
            return 1  # structured integration (MCP / COM / native API)
        # Default to Tier 2 (UIA / Win32) -- the high-reliability path that
        # covers essentially all ordinary Windows apps. Never silently drop an
        # app-only target into the Tier 3 vision scaffold.
        return 2

    def tier_of(self, action):
        tiers = self.policy.get("action_tiers", {})
        for tier, actions in tiers.items():
            if action in actions:
                return tier
        if action in ("read_screen", "read_file", "inspect_window_title"):
            return "passive"
        if action in ("focus_window", "launch_app", "open_app", "input_key",
                      "type_text", "write_file", "click"):
            return "reversible"
        if action in ("execute_cmd", "delete_file", "format_drive"):
            return "privileged"
        return "reversible"

    def _execute_cmd_allowed(self, command):
        allow = self.policy.get("execute_cmd", {}).get("allowlist", [])
        for pat in allow:
            try:
                if re.match(pat, (command or "").strip()):
                    return True
            except Exception:
                pass
        return False

    def _is_fs_mutating(self, command):
        lowered = (command or "").lower()
        return any(t in lowered for t in (
            "del ", "rm ", "rmdir", "rd ", "remove-item", "format ",
            "erase ", "rm -rf", "rmdir "))

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def parse_action(llm_text):
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
        return name

    # -- kill switch support ----------------------------------------------
    def terminate_active(self):
        proc = self._active_process
        if proc is not None and proc.poll() is None:
            if self._kill_job_tree():
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                self._active_process = None
                return True
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._active_process = None
            return True
        self._active_process = None
        return False

    # -- individual handlers (lazy heavy imports) -------------------------
    def _focus_window(self, title):
        import win32gui
        import win32con
        result = {"found": False}

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                wt = win32gui.GetWindowText(hwnd)
                if title and title.lower() in wt.lower():
                    try:
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                    except Exception:
                        pass
                    result["found"] = True
            return True

        win32gui.EnumWindows(_cb, None)
        return "focused: " + title if result["found"] else "no window matched: " + title

    def _launch_app(self, name):
        if name.lower() not in APP_ALLOWLIST:
            return f"rejected: app '{name}' not allow-listed"
        import pywinauto
        pywinauto.Application().start(self._resolve_app_path(name))
        return f"opened: {name}"

    def _input_key(self, sequence):
        import keyboard
        keyboard.send(sequence)
        return f"sent key sequence: {sequence}"

    def _type_text(self, text):
        import keyboard
        keyboard.write(text, delay=0.01)
        return f"typed {len(text)} chars into focused window"

    def _read_file(self, path):
        if not path or not os.path.exists(path):
            return f"rejected: file not found: {path}"
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()[:20000]

    def _write_file(self, path, content):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content or "")
        return f"wrote {len(content or '')} chars to {path}"

    def _read_screen(self):
        try:
            import alison_sense
            return alison_sense.get_visual_context()
        except Exception:
            return "read_screen unavailable"

    def _click(self):
        import pywinauto
        pywinauto.mouse.click()
        return "clicked at cursor"

    def _delete_file(self, path):
        if not path or not os.path.exists(path):
            return f"rejected: file not found: {path}"
        os.remove(path)
        return f"deleted: {path}"

    def _execute_cmd(self, command, confirmed=False, allow=False):
        if not command:
            return "rejected: empty command"
        if not (allow or confirmed):
            return "denied: execute_cmd requires allowlist match or confirmation"
        force_dry = self.global_dry_run
        fs_mut = self._is_fs_mutating(command)
        if force_dry or (fs_mut and not allow):
            preview = f"[DRY-RUN] would execute: {command}"
            if fs_mut:
                preview += " (filesystem-mutating -> confirmation/allowlist required to actually run)"
            return preview
        try:
            jail = self.policy.get("execute_cmd", {}).get(
                "workdir_jail") or tempfile.gettempdir()
            try:
                os.makedirs(jail, exist_ok=True)
            except Exception:
                jail = tempfile.gettempdir()
            creationflags = win32con.CREATE_SUSPENDED if HAS_WIN32_JOB else 0
            proc = subprocess.Popen(
                command, shell=True, cwd=jail,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                creationflags=creationflags)
            # Contain in the job before the process can run: assign while
            # suspended, then resume its primary thread.
            if HAS_WIN32_JOB:
                try:
                    self._assign_to_job(proc)
                finally:
                    self._resume_suspended(proc)
            self._active_process = proc
            out, err = proc.communicate(timeout=120)
            self._active_process = None
            return out if proc.returncode == 0 else (err or "command failed")
        except subprocess.TimeoutExpired:
            if not self._kill_job_tree():
                try:
                    proc.kill()
                except Exception:
                    pass
            self._active_process = None
            return "timeout: command killed"
        except Exception as e:
            self._active_process = None
            return f"error: {e}"

    # -- main entry point -------------------------------------------------
    def execute(self, action_obj, confirmed=False):
        if not self.enabled:
            return "blocked: action executor disabled (kill switch)"
        if not isinstance(action_obj, dict):
            return "rejected: not an object"
        action = action_obj.get("action")
        if action not in ALLOWED_ACTIONS:
            return f"rejected: unknown action '{action}'"
        args = action_obj.get("args", {}) or {}
        tier = self.tier_of(action)

        command = args.get("command", "") if action == "execute_cmd" else ""
        allow = self._execute_cmd_allowed(command) if action == "execute_cmd" else False

        if tier == "privileged" and not (allow or confirmed):
            return "denied: privileged action requires allowlist match or confirmation"
        if tier == "privileged" and not self._verify_policy_signing():
            return "denied: privileged action requires a signed policy (Tier 3)"

        handlers = {
            "open_app": lambda: self._launch_app(args.get("name", "")),
            "launch_app": lambda: self._launch_app(args.get("name", "")),
            "focus_window": lambda: self._focus_window(args.get("title", "")),
            "input_key": lambda: self._input_key(args.get("sequence", args.get("key", ""))),
            "type_text": lambda: self._type_text(args.get("text", "")),
            "read_file": lambda: self._read_file(args.get("path", "")),
            "write_file": lambda: self._write_file(args.get("path", ""), args.get("content", "")),
            "read_screen": lambda: self._read_screen(),
            "click": lambda: self._click(),
            "delete_file": lambda: self._delete_file(args.get("path", "")),
            "execute_cmd": lambda: self._execute_cmd(command, confirmed=confirmed, allow=allow),
        }
        try:
            return handlers[action]()
        except Exception as e:
            return f"error executing {action}: {e}"

    def execute_llm(self, llm_text):
        obj = self.parse_action(llm_text)
        if obj is None:
            return "no action found"
        return self.execute(obj)
