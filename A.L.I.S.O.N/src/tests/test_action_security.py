"""W3.6 verification: ActionExecutor security model (no alison_core import).

Covers: tier resolution, execute_cmd allowlist vs confirmation gating,
kill-switch (enabled flag blocks new actions AND terminates in-flight
subprocess), and basic file I/O actions.
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import alison_actions as aa

EXE = aa.ActionExecutor()


def run_in_thread(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def main():
    ok = True

    # --- tier resolution ---
    if EXE.tier_of("execute_cmd") != "privileged":
        print("FAIL: execute_cmd not privileged"); ok = False
    if EXE.tier_of("read_file") != "passive":
        print("FAIL: read_file not passive"); ok = False
    if EXE.tier_of("focus_window") != "reversible":
        print("FAIL: focus_window not reversible"); ok = False

    # --- capability tier (universal 3-tier) ---
    if EXE.select_capability_tier(app="UnknownApp") != 2:
        print("FAIL: unknown app should resolve to Tier 2 (UIA/Win32)"); ok = False

    # --- execute_cmd: allowlist match executes ---
    r = EXE.execute({"action": "execute_cmd", "args": {"command": "echo ALLOWED"}})
    if "ALLOWED" not in r:
        print(f"FAIL: allowlisted execute_cmd did not run: {r!r}"); ok = False

    # --- execute_cmd: no allowlist + no confirmation -> denied ---
    r = EXE.execute({"action": "execute_cmd", "args": {"command": "echo SECRET"}})
    if "denied" not in r:
        print(f"FAIL: unapproved execute_cmd not denied: {r!r}"); ok = False

    # --- execute_cmd: explicit confirmation runs (non-mutating) ---
    r = EXE.execute({"action": "execute_cmd", "args": {"command": "echo CONFIRMED"}},
                    confirmed=True)
    if "CONFIRMED" not in r:
        print(f"FAIL: confirmed execute_cmd did not run: {r!r}"); ok = False

    # --- unknown action rejected ---
    r = EXE.execute({"action": "hack_the_planet"})
    if "rejected" not in r:
        print(f"FAIL: unknown action not rejected: {r!r}"); ok = False

    # --- kill switch: enabled flag blocks new actions ---
    EXE.enabled = False
    r = EXE.execute({"action": "execute_cmd", "args": {"command": "echo ALLOWED"}})
    if "blocked" not in r:
        print(f"FAIL: disabled executor still ran action: {r!r}"); ok = False
    EXE.enabled = True

    # --- kill switch: terminates an IN-FLIGHT subprocess ---
    def _run():
        return EXE.execute(
            {"action": "execute_cmd", "args": {"command": "ping -n 20 127.0.0.1"}})
    th = run_in_thread(_run)
    threading.Event().wait(1.0)
    alive = EXE._active_process is not None and EXE._active_process.poll() is None
    if not alive:
        print("FAIL: in-flight subprocess not tracked/alive"); ok = False
    killed = EXE.terminate_active()
    th.join(timeout=10)
    still_alive = EXE._active_process is not None and EXE._active_process.poll() is None
    if not killed or still_alive:
        print("FAIL: kill switch did not terminate in-flight subprocess"); ok = False

    # --- file I/O actions ---
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        EXE.execute({"action": "write_file", "args": {"path": path, "content": "hello"}})
        r = EXE.execute({"action": "read_file", "args": {"path": path}})
        if "hello" not in r:
            print(f"FAIL: read_file did not return written content: {r!r}"); ok = False
    finally:
        if os.path.exists(path):
            os.remove(path)

    if ok:
        print("ACTION SECURITY OK: tiers, allowlist/confirmation gating, kill-switch "
              "(block + in-flight terminate), file I/O all verified")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
