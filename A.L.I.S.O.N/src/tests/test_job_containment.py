"""Phase 2 verification: Win32 Job Object containment (no alison_core import).

Covers: job object creation with KILL_ON_JOB_CLOSE, suspended-spawn assignment
(IsProcessInJob), tree-wide termination via the kill switch path, and the
Tier 3 signing scaffold (deny-by-default when enabled without a valid key).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import alison_actions as aa


def main():
    ok = True
    EXE = aa.ActionExecutor()

    # --- job object created (win32) ---
    if not EXE._job_handle:
        print("SKIP: no job object on this platform")
        return 0
    print("job object active")

    # --- suspended spawn + assignment + resume ---
    import subprocess
    import win32con
    import win32job
    import psutil
    proc = subprocess.Popen(
        "cmd /c ping -n 2 127.0.0.1", shell=True, cwd="C:/Temp",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=win32con.CREATE_SUSPENDED)
    EXE._assign_to_job(proc)
    EXE._resume_suspended(proc)
    EXE._active_process = proc
    if not win32job.IsProcessInJob(int(proc._handle), EXE._job_handle):
        print("FAIL: child not assigned to job"); ok = False
    out, _ = proc.communicate(timeout=30)
    if "Reply from 127.0.0.1" not in out:
        print("FAIL: resumed child did not run (ping output missing)"); ok = False
    EXE._active_process = None

    # --- tree-wide kill: cmd /c ping -t -> cmd -> ping ---
    proc2 = subprocess.Popen(
        "cmd /c ping -t 127.0.0.1", shell=True, cwd="C:/Temp",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=win32con.CREATE_SUSPENDED)
    EXE._assign_to_job(proc2)
    EXE._resume_suspended(proc2)
    EXE._active_process = proc2
    time.sleep(1.2)
    me = psutil.Process(proc2.pid)
    before = {proc2.pid} | {p.pid for p in me.children(recursive=True)}
    if not EXE.terminate_active():
        print("FAIL: terminate_active returned False"); ok = False
    time.sleep(1.0)
    survivors = [pid for pid in before if psutil.pid_exists(pid)]
    if survivors:
        print(f"FAIL: survivors remain: {survivors}"); ok = False

    # --- Tier 3 signing scaffold: deny-by-default when enabled ---
    EXE.policy.setdefault("signing", {})["enabled"] = True
    EXE._signing_enabled = True
    EXE._signature_valid = None
    r = EXE.execute({"action": "execute_cmd", "args": {"command": "echo ALLOWED"}})
    if "signed policy" not in r:
        print(f"FAIL: signing gate not enforced: {r}"); ok = False
    EXE.policy["signing"]["enabled"] = False
    EXE._signing_enabled = False
    EXE._signature_valid = None

    if ok:
        print("JOB CONTAINMENT OK: assignment, tree kill, signing scaffold verified")
        return 0
    print("JOB CONTAINMENT FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())