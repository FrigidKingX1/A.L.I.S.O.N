"""alison_hw.py -- hardware telemetry via nvml.dll (ctypes, zero pip deps).

NVML ships with every NVIDIA driver (System32\\nvml.dll), so no new Python
dependency or PyInstaller bundle change is needed. Publishes 1 Hz `hardware`
events over the AlisonIPC PUB socket: gpu_util_pct, vram_used_mb,
vram_total_mb, gpu_temp_c. Degrades gracefully to None when the DLL, the GPU,
or the driver is unavailable -- telemetry is optional, never fatal.
"""
import ctypes
import threading

NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _NvmlProbe:
    """Lazy NVML binding. All calls are guarded; a failed init poisons
    nothing -- every public method returns None."""

    def __init__(self):
        self._lib = None
        self._handle = None
        self._init()

    def _init(self):
        try:
            self._lib = ctypes.WinDLL("nvml.dll")
        except Exception:
            return
        try:
            self._lib.nvmlInit_v2.restype = ctypes.c_int
            self._lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            self._lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            self._lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            self._lib.nvmlDeviceGetTemperature.restype = ctypes.c_int
            self._lib.nvmlShutdown.restype = ctypes.c_int
            if self._lib.nvmlInit_v2() != NVML_SUCCESS:
                return
            handle = ctypes.c_void_p()
            if self._lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != NVML_SUCCESS:
                return
            self._handle = handle
        except Exception:
            self._handle = None

    def read(self):
        """Return dict(gpu_util_pct, vram_used_mb, vram_total_mb, gpu_temp_c)
        or None when NVML/GPU is unavailable."""
        if self._handle is None:
            return None
        try:
            util = _NvmlUtilization()
            if self._lib.nvmlDeviceGetUtilizationRates(
                    self._handle, ctypes.byref(util)) != NVML_SUCCESS:
                return None
            mem = _NvmlMemory()
            if self._lib.nvmlDeviceGetMemoryInfo(
                    self._handle, ctypes.byref(mem)) != NVML_SUCCESS:
                return None
            temp = ctypes.c_uint()
            if self._lib.nvmlDeviceGetTemperature(
                    self._handle, NVML_TEMPERATURE_GPU, ctypes.byref(temp)) != NVML_SUCCESS:
                temp = ctypes.c_uint(0)
            return {
                "gpu_util_pct": int(util.gpu),
                "vram_used_mb": int(mem.used // (1024 * 1024)),
                "vram_total_mb": int(mem.total // (1024 * 1024)),
                "gpu_temp_c": int(temp.value),
            }
        except Exception:
            return None


_probe = None
_probe_lock = threading.Lock()


def read_once():
    """Thread-safe one-shot hardware sample (used by get_status)."""
    global _probe
    with _probe_lock:
        if _probe is None:
            _probe = _NvmlProbe()
        return _probe.read()


def start_hardware_monitor(ipc, stop_ev, interval=1.0):
    """Daemon thread: publish 1 Hz `hardware` events. Never raises."""
    while not stop_ev.is_set():
        try:
            sample = read_once()
            if ipc is not None and sample is not None:
                ipc.publish_event("hardware", sample)
        except Exception:
            pass
        stop_ev.wait(interval)