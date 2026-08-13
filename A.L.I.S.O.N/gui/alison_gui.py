"""A.L.I.S.O.N. GUI host.

Runs as a separate, lightweight process from the core (alison_core.py).
Connects over IPC (shared-memory telemetry + ZeroMQ event/control) and
renders the ambient HUD: a dashboard (brain radar, control panel,
hippocampal view) plus a frameless ambient overlay with a GLSL arc
visualizer. Includes a system tray and global hotkeys.
"""

import os
import sys
import time
import threading

# Allow importing the engine-side bridge (alison_ipc) from the engine source
# tree (A.L.I.S.O.N/src) regardless of the current working directory.
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt6.QtCore import (
    QObject, QThread, pyqtSignal, pyqtProperty, pyqtSlot, QUrl, Qt, QTimer, QSize,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtQuick import QQuickView

import alison_ipc
from theme import (
    APP_NAME, CYAN, VIOLET, BG, BG_SOFT, TEXT, MUTED, DRIVE_LABELS, PROACTIVE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
QML_DIR = os.path.join(HERE, "qml")
CORE_LAUNCH_CMD = os.environ.get("ALISON_CORE_CMD")


# ----------------------------------------------------------------------
# Bridge: shared state between Python host and QML
# ----------------------------------------------------------------------
class Bridge(QObject):
    telemetryUpdated = pyqtSignal()
    eventReceived = pyqtSignal(str, str)        # topic, text
    overlayRequested = pyqtSignal(bool)         # show?
    statusChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gamma = 1.0
        self._drives = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._affect = [0.0] * alison_ipc.AFFECT_DIM
        self._activity = 0.0
        self._status = "initialising"
        self._core_online = False
        self._core_proc = None
        self._pop_count = 0
        self._listening = False

    # --- properties exposed to QML ---
    @pyqtProperty(float, notify=telemetryUpdated)
    def gamma(self):
        return self._gamma

    @pyqtProperty("QVariantList", notify=telemetryUpdated)
    def drives(self):
        return self._drives

    @pyqtProperty("QVariantList", notify=telemetryUpdated)
    def affect(self):
        return self._affect

    @pyqtProperty(float, notify=telemetryUpdated)
    def activity(self):
        return self._activity

    @pyqtProperty(str, notify=statusChanged)
    def status(self):
        return self._status

    @status.setter
    def status(self, v):
        if v != self._status:
            self._status = v
            self.statusChanged.emit()

    @pyqtProperty(bool, notify=statusChanged)
    def coreOnline(self):
        return self._core_online

    @coreOnline.setter
    def coreOnline(self, v):
        if v != self._core_online:
            self._core_online = v
            self.statusChanged.emit()

    @pyqtProperty(bool, notify=statusChanged)
    def listening(self):
        return self._listening

    @listening.setter
    def listening(self, v):
        if v != self._listening:
            self._listening = v
            self.statusChanged.emit()

    # --- called from worker threads ---
    def push_telemetry(self, affect, gamma, drives):
        self._gamma = gamma
        self._drives = [float(x) for x in drives]
        self._affect = [float(x) for x in affect]
        self.telemetryUpdated.emit()

    def bump_activity(self, amt=1.0):
        self._activity = min(1.0, self._activity + amt)

    def decay_activity(self):
        if self._activity > 0.0:
            self._activity = max(0.0, self._activity - 0.06)
            self.telemetryUpdated.emit()

    def evaluate_proactive(self):
        """Drive the proactive auto-pop from current telemetry."""
        g, ax, gu = self._gamma, self._drives[2], self._drives[4]
        breach = (g >= PROACTIVE["gamma"] or
                  ax >= PROACTIVE["anxiety"] or
                  gu >= PROACTIVE["goal_urgency"])
        if breach:
            self._pop_count += 1
        else:
            self._pop_count = 0
        if self._pop_count >= PROACTIVE["frames"]:
            self._pop_count = 0
            self.overlayRequested.emit(True)

    # --- QML-invokable control ---
    @pyqtSlot(str, "QVariant")
    def sendCommand(self, action, params=None):
        try:
            cctx, creq = alison_ipc.AlisonIPC.make_control()
            msg = {"cmd": action}
            if isinstance(params, dict):
                msg.update(params)
            creq.send_json(msg)
            reply = creq.recv_json()
            creq.close()
            cctx.term()
            self.eventReceived.emit("control", f"{action} -> {reply}")
        except Exception as exc:
            self.eventReceived.emit("control", f"{action} ERROR: {exc}")

    def _core_command(self):
        """Resolve how to start the engine.

        Frozen install: ``ALISON_Core.exe`` living next to this executable.
        Dev: honour ``ALISON_CORE_CMD`` (e.g. ``python``), else fall back to
        ``python alison_core.py``. Returns an argument list (never a shell
        string) so paths with spaces (e.g. ``C:\\Program Files\\A.L.I.S.O.N.\\``)
        are passed correctly.
        """
        if getattr(sys, "frozen", False):
            exe = os.path.join(os.path.dirname(sys.executable), "ALISON_Core.exe")
            if os.path.exists(exe):
                return [exe, "--ipc", "--auto"]
        env = CORE_LAUNCH_CMD
        if env:
            return env.split() + ["alison_core.py", "--ipc", "--auto"]
        return [sys.executable, os.path.join(ROOT, "alison_core.py"), "--ipc", "--auto"]

    @pyqtSlot()
    def launchCore(self):
        # Single-instance guard: attach to an already-running service instead
        # of spawning a duplicate (autostart / GUI relaunch scenario).
        if self._core_online:
            self.eventReceived.emit("control", "Core already online; reusing instance.")
            return
        try:
            import subprocess
            cmd = self._core_command()
            log_path = os.path.join(os.path.dirname(sys.executable), "alison_core.out.log")
            self._core_log = open(log_path, "ab", buffering=0)
            self._core_proc = subprocess.Popen(
                cmd, stdout=self._core_log, stderr=self._core_log)
            self.eventReceived.emit(
                "control", "Core launch requested: " + " ".join(cmd) +
                " | log: " + log_path)
        except Exception as exc:
            self._core_proc = None
            self.eventReceived.emit("control", f"launch failed: {exc}")

    @pyqtSlot(int, int)
    def moveOverlay(self, dx, dy):
        view = getattr(self, "_overlay_view", None)
        if view is not None:
            pos = view.position()
            view.setPosition(pos.x() + dx, pos.y() + dy)

    @pyqtSlot()
    def toggleOverlay(self):
        view = getattr(self, "_overlay_view", None)
        if view is not None:
            if view.isVisible():
                view.hide()
            else:
                view.show()
                view.raise_()


# ----------------------------------------------------------------------
# Telemetry reader thread
# ----------------------------------------------------------------------
class TelemetryThread(QThread):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                affect, gamma, drives = alison_ipc.AlisonIPC.read_telemetry_full()
                self.bridge.push_telemetry(affect, gamma, drives)
                self.bridge.coreOnline = True
            except FileNotFoundError:
                self.bridge.coreOnline = False
            except Exception:
                self.bridge.coreOnline = False
            self.bridge.evaluate_proactive()
            time.sleep(1.0 / 60.0)

    def stop(self):
        self._stop.set()


# ----------------------------------------------------------------------
# Event subscriber thread
# ----------------------------------------------------------------------
class EventThread(QThread):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                ctx, sub = alison_ipc.AlisonIPC.make_subscriber()
            except Exception:
                time.sleep(1.0)
                continue
            try:
                while not self._stop.is_set():
                    if sub.poll(500):
                        topic, payload = sub.recv_multipart()
                        self._dispatch(topic.decode(), payload.decode())
                    else:
                        break  # slow-joiner / publisher gone; resubscribe
            except Exception:
                pass
            finally:
                try:
                    sub.close()
                    ctx.term()
                except Exception:
                    pass
            time.sleep(0.5)

    def _dispatch(self, topic, payload):
        import json
        try:
            data = json.loads(payload)
        except Exception:
            data = {}
        if topic == "token_stream":
            text = data.get("text", "")
            self.bridge.bump_activity(0.5)
            self.bridge.eventReceived.emit("token", text)
        elif topic == "screen_context":
            self.bridge.eventReceived.emit("screen", str(data.get("context", "")))
        elif topic == "thought":
            self.bridge.eventReceived.emit("thought", str(data))
        elif topic == "log":
            self.bridge.eventReceived.emit("log", str(data))
        elif topic == "ear_state":
            state = (data.get("state") if isinstance(data, dict) else str(data))
            self.bridge.listening = (state == "listening")
            self.bridge.eventReceived.emit("ear", str(data))
        else:
            self.bridge.eventReceived.emit(topic, str(data))

    def stop(self):
        self._stop.set()


# ----------------------------------------------------------------------
# GUI assembly
# ----------------------------------------------------------------------
def load_view(qml_file, bridge, frameless=False, w=420, h=420):
    view = QQuickView()
    view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
    view.rootContext().setContextProperty("bridge", bridge)
    view.rootContext().setContextProperty("THEME", {
        "cyan": CYAN, "violet": VIOLET, "bg": BG, "bgSoft": BG_SOFT,
        "text": TEXT, "muted": MUTED, "appName": APP_NAME,
        "driveLabels": DRIVE_LABELS,
    })
    view.setSource(QUrl.fromLocalFile(os.path.join(QML_DIR, qml_file)))
    if frameless:
        view.setFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
    view.setMinimumSize(QSize(w, h))
    return view


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)

    bridge = Bridge()

    # Dashboard
    dashboard = load_view("main.qml", bridge, w=1100, h=720)
    dashboard.setTitle(APP_NAME)

    # Ambient overlay (frameless, hidden until invoked)
    overlay = load_view("AmbientOverlay.qml", bridge, frameless=True, w=420, h=420)
    bridge._overlay_view = overlay
    overlay.hide()

    def show_overlay(show):
        if show:
            overlay.show()
            overlay.raise_()
        else:
            overlay.hide()

    bridge.overlayRequested.connect(show_overlay)

    # Tray
    tray = QSystemTrayIcon(QIcon(), app)
    tray.setToolTip(APP_NAME)
    menu = QMenu()
    act_show = menu.addAction("Show Dashboard")
    act_overlay = menu.addAction("Toggle Ambient Overlay")
    act_launch = menu.addAction("Launch Core")
    menu.addSeparator()
    act_quit = menu.addAction("Quit")
    act_show.triggered.connect(dashboard.show)
    act_overlay.triggered.connect(lambda: show_overlay(not overlay.isVisible()))
    act_launch.triggered.connect(bridge.launchCore)
    act_quit.triggered.connect(app.quit)
    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: dashboard.show()
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None
    )
    tray.show()

    # Worker threads
    t_thread = TelemetryThread(bridge)
    e_thread = EventThread(bridge)
    t_thread.start()
    e_thread.start()

    # Auto-launch the engine as a persistent service. The 1.5s delay lets the
    # IPC thread first detect an already-running Core (autostart / GUI relaunch)
    # so we attach instead of spawning a duplicate instance.
    QTimer.singleShot(1500, bridge.launchCore)

    # Activity decay timer (drives the voice-waveform pulse)
    decay = QTimer()
    decay.setInterval(60)
    decay.timeout.connect(bridge.decay_activity)
    decay.start()

    # Watchdog: if the Core drops (crash) and no process is alive, relaunch it
    # so the ambient copilot self-heals instead of staying brain-dead. It caps
    # at a few attempts so a persistently-failing Core (e.g. missing model
    # weights) does not spawn an endless relaunch storm the user cannot escape.
    bridge._core_failures = 0
    watchdog = QTimer()
    watchdog.setInterval(5000)
    def _watchdog():
        proc = bridge._core_proc
        alive = proc is not None and proc.poll() is None
        if bridge.coreOnline:
            bridge._core_failures = 0
            return
        if alive:
            return
        bridge._core_failures += 1
        if bridge._core_failures > 4:
            bridge.eventReceived.emit(
                "control",
                "Core failed to start after several attempts -- see alison_core.out.log")
            return
        bridge.launchCore()
    watchdog.timeout.connect(_watchdog)
    watchdog.start()

    # Clean shutdown: terminate the Core child process when the GUI exits so it
    # does not linger (and keep spawning consoles) after the window closes.
    def _cleanup_core():
        proc = getattr(bridge, "_core_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    app.aboutToQuit.connect(_cleanup_core)

    # Global hotkeys
    import keyboard
    try:
        keyboard.add_hotkey("alt+space", lambda: show_overlay(not overlay.isVisible()))
        keyboard.add_hotkey("ctrl+alt+a", dashboard.show)
        keyboard.add_hotkey("alt+v", lambda: bridge.sendCommand("start_listen"))
        keyboard.add_hotkey("ctrl+alt+k", lambda: bridge.sendCommand("kill_switch"))
    except Exception:
        pass

    bridge.status = "online" if bridge.coreOnline else "connecting to core..."
    dashboard.show()

    exit_code = app.exec()
    t_thread.stop()
    e_thread.stop()
    t_thread.wait(1000)
    e_thread.wait(1000)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
