"""
Cross-process integration test for the Aether engine IPC layer.
Starts alison_core.py --ipc, then (from a separate process context)
verifies, using the real IPC channels:
  - shared-memory telemetry is readable (128-dim affect + gamma)
  - the ZeroMQ control channel round-trips commands
  - the event stream delivers 'log' events
Readiness is detected by probing the telemetry shared memory directly
(rather than scraping the engine's stdout, which the subprocess holds
open for writing).
"""
import subprocess, sys, os, time, json, threading

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\dgc12\AppData\Local\Programs\Python\Python310\python.exe"
ENGINE = os.path.join(HERE, "alison_core.py")

sys.path.insert(0, HERE)
import alison_ipc  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402


def _clean_stale_shm():
    try:
        from multiprocessing import shared_memory
        s = shared_memory.SharedMemory(name=alison_ipc.SHM_NAME)
        s.close()
        s.unlink()
    except Exception:
        pass


def main():
    _clean_stale_shm()
    proc = subprocess.Popen([PY, "-u", ENGINE, "--ipc"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            cwd=HERE)

    def _watchdog():
        try:
            proc.kill()
        except Exception:
            pass
        os._exit(2)

    _wd = threading.Timer(280, _watchdog)
    _wd.daemon = True
    _wd.start()

    # 1) wait for IPC by probing the telemetry shared memory
    online = False
    t0 = time.time()
    while time.time() - t0 < 240:
        try:
            a, g = alison_ipc.AlisonIPC.read_telemetry()
            if a.shape == (128,) and np.isfinite(a).all() and np.isfinite(g):
                online = True
                break
        except Exception:
            pass
        if proc.poll() is not None:
            print("ENGINE EXITED before IPC online")
            return 1
        time.sleep(2)
    if not online:
        print("FAIL: telemetry never available (engine IPC not online in 240s)")
        proc.kill()
        return 1
    print("[ok] engine IPC online (telemetry readable)")

    # 2) shared-memory telemetry stability
    tele = []
    for _ in range(15):
        try:
            a, g = alison_ipc.AlisonIPC.read_telemetry()
            tele.append((float(np.linalg.norm(a)), float(g)))
        except Exception as e:
            tele.append(("ERR", str(e)))
        time.sleep(0.15)
    ok_tele = sum(1 for t in tele if isinstance(t[0], float))
    print(f"[telemetry] {ok_tele}/15 reads ok; sample={tele[0] if tele else None}")

    # 3) control channel round-trips
    cctx, creq = alison_ipc.AlisonIPC.make_control()
    creq.setsockopt(zmq.RCVTIMEO, 4000)
    replies = {}
    for cmd in ({"cmd": "get_status"},
                {"cmd": "set_gamma_bounds", "low": 0.2, "high": 1.5},
                {"cmd": "toggle_screen_sense"},
                {"cmd": "get_status"}):
        try:
            creq.send_json(cmd)
            replies[cmd["cmd"]] = creq.recv_json()
        except Exception as e:
            replies[cmd["cmd"]] = {"ok": False, "error": str(e)}
    st = replies.get("get_status", {})
    gb = replies.get("set_gamma_bounds", {})
    ts = replies.get("toggle_screen_sense", {})
    st2 = replies.get("get_status", {})
    print(f"[control] get_status={st.get('ok')} set_gamma={gb} toggle={ts} "
          f"gamma_bounds_after={st2.get('gamma_bounds')}")

    # 4) event stream (subscribe)
    ctx, sub = alison_ipc.AlisonIPC.make_subscriber()
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    events = {}
    t0 = time.time()
    while time.time() - t0 < 8:
        socks = dict(poller.poll(300))
        if sub in socks:
            try:
                topic, data = sub.recv_multipart(flags=zmq.NOBLOCK)
                events[topic.decode()] = events.get(topic.decode(), 0) + 1
            except Exception:
                break
    print(f"[events] received: {events}")

    # 5) verdict
    failures = []
    if ok_tele < 10:
        failures.append("telemetry unreadable")
    if not (st.get("ok") and gb.get("ok") and ts.get("ok")):
        failures.append("control command failed")
    if st2.get("gamma_bounds") != [0.2, 1.5]:
        failures.append(f"gamma bounds not applied: {st2.get('gamma_bounds')}")
    if not events.get("log"):
        failures.append("no 'log' events received")

    proc.kill()
    try:
        sub.close(0)
        creq.close(0)
        cctx.term()
        ctx.term()
    except Exception:
        pass

    if failures:
        print("INTEGRATION FAIL:", failures)
        return 1
    print("INTEGRATION PASS: telemetry + control + event stream all verified cross-process")
    return 0


if __name__ == "__main__":
    sys.exit(main())
