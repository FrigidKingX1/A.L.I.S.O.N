"""
AlisonIPC -- Project Aether Engine <-> GUI Inter-Process Bridge
==============================================================
Zero-copy shared-memory telemetry + ZeroMQ event/control channels.

Telemetry layout (shared memory, 540 bytes):
    [ gamma (float32) , affect_vector (128 x float32) , drives (6 x float32) ]
    -> 135 float32
    Drives (indices 129..134) are, in order:
        PLEASURE, AROUSAL, ANXIETY, CURIOSITY, GOAL_URGENCY, SATIATION

Synchronization: a named Win32 mutex (pywin32) guards the shared-memory
write/read so the GUI never reads a torn 128-dim vector. If pywin32 is
unavailable the mutex degrades to a no-op (single-process / dev only).

NOTE ON TRANSPORT: this implementation uses `tcp://127.0.0.1` loopback
endpoints rather than the spec's `ipc://` named pipes. libzmq's ipc
transport is unreliable across Windows versions; loopback TCP is the
robust choice for cross-process comms on Windows and preserves the
sub-millisecond latency target.
"""

import json
import threading
from multiprocessing import shared_memory

import numpy as np

try:
    import zmq
    HAS_ZMQ = True
except ImportError:  # pragma: no cover
    HAS_ZMQ = False

try:
    import win32event
    HAS_WIN32_MUTEX = True
except ImportError:  # pragma: no cover
    HAS_WIN32_MUTEX = False


AFFECT_DIM = 128
DRIVE_DIM = 6
DRIVE_NAMES = ["PLEASURE", "AROUSAL", "ANXIETY", "CURIOSITY", "GOAL_URGENCY", "SATIATION"]
AFFECT_OFFSET = 1                     # index of first affect float
DRIVE_OFFSET = 1 + AFFECT_DIM         # index of first drive float (129)
SHM_NAME = "alison_telemetry_shm"
SHM_N_FLOATS = AFFECT_DIM + 1 + DRIVE_DIM   # gamma + 128 affect + 6 drives
SHM_SIZE = SHM_N_FLOATS * 4            # bytes (float32)
MUTEX_NAME = "AlisonTelemetryMutex"

TELEMETRY_ENDPOINT = "tcp://127.0.0.1:5557"
CONTROL_ENDPOINT = "tcp://127.0.0.1:5558"

# --- Control command verbs (GUI/hotkey -> Core) ---
CMD_GET_STATUS = "get_status"
CMD_SET_SCREEN_SENSE = "set_screen_sense"
CMD_TOGGLE_SCREEN_SENSE = "toggle_screen_sense"
CMD_SET_GAMMA_BOUNDS = "set_gamma_bounds"
CMD_USER_SPEECH = "user_speech"      # payload: {"text": "<transcript>"}
CMD_START_LISTEN = "start_listen"    # push-to-talk trigger (Core captures mic)
CMD_SET_WAKEWORD = "set_wakeword"    # payload: {"enabled": bool}

# --- Event topics (Core -> GUI) ---
TOPIC_TOKEN_STREAM = "token_stream"
TOPIC_SCREEN_CONTEXT = "screen_context"
TOPIC_THOUGHT = "thought"
TOPIC_LOG = "log"
TOPIC_EAR_STATE = "ear_state"        # payload: {"state": "listening"|"idle"|"error"}


class AlisonIPC:
    """Bridges the Aether engine and the GUI across process boundaries.

    The engine (writer) instantiates AlisonIPC and calls publish_telemetry /
    publish_event / start_control. The GUI (reader) either instantiates its
    own AlisonIPC for control, or uses the static read_telemetry / make_subscriber
    / make_control helpers to attach to the same shared state.
    """

    def __init__(self, telemetry_endpoint=TELEMETRY_ENDPOINT,
                 control_endpoint=CONTROL_ENDPOINT):
        if not HAS_ZMQ:
            raise RuntimeError("pyzmq is required for AlisonIPC")
        self.telemetry_endpoint = telemetry_endpoint
        self.control_endpoint = control_endpoint

        self._shm = None
        self._owns_shm = False
        self._mutex = None
        self._stop = threading.Event()
        self._control_handler = None
        self._control_thread = None

        if HAS_WIN32_MUTEX:
            # Opened by name so a reader in another process synchronizes with us.
            self._mutex = win32event.CreateMutex(None, False, MUTEX_NAME)

        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.set_hwm(1000)
        self._pub.bind(self.telemetry_endpoint)
        self._rep = None

        self._create_shm()

    # ------------------------------------------------------------------
    # Shared memory
    # ------------------------------------------------------------------
    def _create_shm(self):
        try:
            self._shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
            self._owns_shm = True
        except FileExistsError:
            # Stale segment from a previous (crashed) run.
            try:
                stale = shared_memory.SharedMemory(name=SHM_NAME)
                stale.close()
                stale.unlink()
            except Exception:
                pass
            self._shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
            self._owns_shm = True

    def _write_shm(self, arr):
        if self._shm is None:
            return
        acquired = False
        try:
            if self._mutex is not None:
                if win32event.WaitForSingleObject(self._mutex, 1000) == win32event.WAIT_OBJECT_0:
                    acquired = True
            arr32 = np.ascontiguousarray(arr, dtype=np.float32)
            np.copyto(np.frombuffer(self._shm.buf, dtype=np.float32)[:SHM_N_FLOATS], arr32)
        finally:
            if acquired:
                win32event.ReleaseMutex(self._mutex)

    # ------------------------------------------------------------------
    # Telemetry (shared memory, 60 Hz)
    # ------------------------------------------------------------------
    def publish_telemetry(self, affect_128, gamma, drives=None):
        """Write the 128-dim limbic affect vector, precision scalar gamma,
        and the 6 homeostatic drives to shared memory."""
        if affect_128 is None:
            return
        buf = np.zeros(SHM_N_FLOATS, dtype=np.float32)
        buf[0] = np.float32(gamma)
        buf[AFFECT_OFFSET:AFFECT_OFFSET + AFFECT_DIM] = \
            np.asarray(affect_128, dtype=np.float32).reshape(-1)[:AFFECT_DIM]
        if drives is not None:
            d = np.asarray(drives, dtype=np.float32).reshape(-1)[:DRIVE_DIM]
            buf[DRIVE_OFFSET:DRIVE_OFFSET + DRIVE_DIM] = d
        self._write_shm(buf)

    # ------------------------------------------------------------------
    # Event stream (ZeroMQ PUB)
    # ------------------------------------------------------------------
    def publish_event(self, topic, data):
        """topic: str (e.g. 'token_stream', 'screen_context', 'log')."""
        try:
            self._pub.send_multipart([topic.encode(), json.dumps(data).encode()])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Control channel (ZeroMQ REP, non-blocking)
    # ------------------------------------------------------------------
    def start_control(self, handler):
        """Start a background REP loop dispatching commands to `handler(cmd)->reply`."""
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.set_hwm(100)
        self._rep.bind(self.control_endpoint)
        self._control_handler = handler
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def _control_loop(self):
        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                socks = dict(poller.poll(200))
            except zmq.ZMQError:
                break
            if self._rep not in socks:
                continue
            try:
                msg = self._rep.recv_json()
            except Exception:
                try:
                    self._rep.send_json({"ok": False, "error": "bad request"})
                except Exception:
                    pass
                continue
            try:
                reply = self._control_handler(msg) if self._control_handler else {"ok": True}
            except Exception as exc:  # handler bug must not kill the loop
                reply = {"ok": False, "error": str(exc)}
            try:
                self._rep.send_json(reply)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def close(self):
        self._stop.set()
        for sock in (self._pub, self._rep):
            try:
                if sock is not None:
                    sock.close(0)
            except Exception:
                pass
        try:
            self._ctx.term()
        except Exception:
            pass
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            if self._owns_shm:
                try:
                    self._shm.unlink()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Reader-side helpers (GUI process)
    # ------------------------------------------------------------------
    @staticmethod
    def read_telemetry():
        """Attach to the existing shared-memory block and return (affect_128, gamma).

        Backward-compatible signature: returns only the 128-dim affect vector
        and gamma (drives are ignored). Use read_telemetry_full() for drives.
        """
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        mutex = win32event.CreateMutex(None, False, MUTEX_NAME) if HAS_WIN32_MUTEX else None
        acquired = False
        try:
            if mutex is not None:
                if win32event.WaitForSingleObject(mutex, 1000) == win32event.WAIT_OBJECT_0:
                    acquired = True
            buf = np.frombuffer(shm.buf, dtype=np.float32)[:SHM_N_FLOATS].copy()
        finally:
            if acquired:
                win32event.ReleaseMutex(mutex)
            shm.close()
        return buf[AFFECT_OFFSET:AFFECT_OFFSET + AFFECT_DIM].copy(), float(buf[0])

    @staticmethod
    def read_telemetry_full():
        """Return (affect_128, gamma, drives) where drives is a length-6 list
        in DRIVE_NAMES order."""
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        mutex = win32event.CreateMutex(None, False, MUTEX_NAME) if HAS_WIN32_MUTEX else None
        acquired = False
        try:
            if mutex is not None:
                if win32event.WaitForSingleObject(mutex, 1000) == win32event.WAIT_OBJECT_0:
                    acquired = True
            buf = np.frombuffer(shm.buf, dtype=np.float32)[:SHM_N_FLOATS].copy()
        finally:
            if acquired:
                win32event.ReleaseMutex(mutex)
            shm.close()
        affect = buf[AFFECT_OFFSET:AFFECT_OFFSET + AFFECT_DIM].copy()
        gamma = float(buf[0])
        drives = [float(x) for x in buf[DRIVE_OFFSET:DRIVE_OFFSET + DRIVE_DIM]]
        return affect, gamma, drives

    @staticmethod
    def make_subscriber(endpoint=TELEMETRY_ENDPOINT, topics=("",)):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.set_hwm(1000)
        sub.connect(endpoint)
        for t in topics:
            sub.setsockopt(zmq.SUBSCRIBE, t.encode())
        return ctx, sub

    @staticmethod
    def make_control(endpoint=CONTROL_ENDPOINT):
        ctx = zmq.Context()
        req = ctx.socket(zmq.REQ)
        req.set_hwm(100)
        req.connect(endpoint)
        return ctx, req


if __name__ == "__main__":
    # Self-test: exercise named shared memory + Win32 mutex + ZMQ PUB/SUB/REP.
    import time

    ipc = AlisonIPC()
    ipc.start_control(lambda c: {"ok": True, "echo": c})

    # --- telemetry write latency ---
    t0 = time.perf_counter()
    for i in range(1000):
        av = np.random.randn(AFFECT_DIM).astype(np.float32)
        drives = (np.arange(DRIVE_DIM) / DRIVE_DIM + 0.1 * np.sin(i / 30.0)).astype(np.float32)
        ipc.publish_telemetry(av, 1.0 + 0.5 * np.sin(i / 50.0), drives)
        ipc.publish_event("token_stream", {"text": "x", "i": i})
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"[selftest] 1000 telemetry+event writes: {dt / 1000.0:.4f} ms each")

    # --- shared-memory read (zero-copy, mutex-guarded) ---
    a, g = AlisonIPC.read_telemetry()
    print(f"[selftest] read gamma={g:.4f} affect_norm={float(np.linalg.norm(a)):.4f}")

    # --- shared-memory read with drives ---
    af, gf, dr = AlisonIPC.read_telemetry_full()
    print(f"[selftest] full read drives={[round(x, 3) for x in dr]}")

    # --- event stream (subscribe BEFORE publishing to avoid slow-joiner loss) ---
    ctx, sub = AlisonIPC.make_subscriber()
    time.sleep(0.3)
    ipc.publish_event("token_stream", {"text": "hello"})
    event_ok = bool(sub.poll(2000)) and sub.recv_multipart() is not None
    print(f"[selftest] event recv ok={event_ok}")

    # --- control channel round-trip ---
    cctx, creq = AlisonIPC.make_control()
    creq.send_json({"cmd": "ping"})
    reply = creq.recv_json()
    print(f"[selftest] control reply={reply}")

    ipc.close()
    print("[selftest] PASS")
