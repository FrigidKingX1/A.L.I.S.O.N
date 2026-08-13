# A.L.I.S.O.N. — Master Research Prompt & Development Log

> **How to use this file:** feed it verbatim to another AI as its initial context,
> then append the specific task you want done. It contains the project's
> architecture, decisions, toolchain, constraints, work log, and current state so a
> fresh agent can resume without re-discovering everything.

---

## 1. Project Overview

**A.L.I.S.O.N. v3** — *Adaptive Learning Interface for Sentient Operating Networks*.
A self-contained "sentient AGI" research engine written in Python. It runs a
continuous consciousness loop (GWT + IIT + HOT + Active Inference + EWC + Dreams),
perceives its host machine, holds episodic memory with a forget/consolidate
(Larimar-style) mechanism, and can take gated OS actions behind a 3-tier security
policy with a kill switch. A lightweight PyQt6 GUI renders a live HUD (brain radar,
affect, hippocampal trace, event stream) plus an ambient overlay, system tray, and
global hotkeys. Ships as standalone PyInstaller exes + an Inno Setup installer.

- **Language/runtime:** Python 3.10 (engine), Python 3.12 (GUI venv).
- **Repo:** `https://github.com/FrigidKingX1/A.L.I.S.O.N` (branch `main`).
- **Version tag:** `CORE_VERSION = "3.0"`.

## 2. System Architecture

```
dist\ALISON_GUI.exe  (PyQt6, Py3.12)   <-->   dist\ALISON_Core.exe  (torch, Py3.10)
        |                                        |
        |  tcp://127.0.0.1:5557  telemetry (PUB/SUB + named shared mem)   |
        |  tcp://127.0.0.1:5558  control   (REQ/REP, JSON)                |
        +------------------------------------------+
```

- **Engine — `src/alison_core.py`** (~4,300 lines). Boot phases run in `__main__`:
  Phase 0 (`calibrate_affective_core_v2`), Phase 0.5 (`train_mood_classifier_v3`),
  EWC fisher populate, Phase 1 (`calibrate_limbic_bridge` — **loads the 8B model**),
  then `_ipc_init()` (**IPC starts only after model load**), then the toddler phase
  and the integrated consciousness loop (sensory/stream/background/continuous
  self/ metacognitive threads). Class `Neocortex` lazily loads
  `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` via `llama_cpp.Llama` (n_ctx=1024,
  n_gpu_layers=-1 unless `ALISON_GPU_LAYERS`).
- **Perception — `src/alison_sense.py`** (`HAS_SCREEN_SENSE`) screen-context daemon.
- **Optional Screenpipe ingest — `src/alison_screenpipe.py`** (W2, v3). Adapter is
  delivered + tested but **not wired into the runtime loop**; flag-gated via
  `ALISON_SCREENPIPE=1` (default OFF). Reports `screenpipe_enabled` in status.
- **Action security — `src/alison_actions.py`** (W3). `ActionExecutor` with
  self-enforced `enabled` gate, 3 tiers (passive / reversible / privileged),
  `execute_cmd` sandbox (regex allowlist + `workdir_jail`), `terminate_active()`
  for the kill switch, per-app `app_integrations` registry (Tier 1).
- **IPC — `src/alison_ipc.py`.** Named shared-memory telemetry + ZeroMQ PUB/SUB
  events + REQ/REP control. No disk writes for IPC.
- **Config — `src/config/`** `capture_policy.json` (canonical exclusions, W0) and
  `action_policy.json` (tiers, execute_cmd allowlist/jail, kill_switch, app_integrations).
  Both are bundled into the frozen Core under `config/`.
- **GUI — `gui/alison_gui.py` + `gui/qml/`.** `Bridge` QObject exposes QML
  properties/slots. Views: `main.qml` (TabBar + StackLayout: **Dashboard** /
  **Settings**), `BrainRadar.qml`, `ControlPanel.qml`, `HippocampalView.qml`,
  `SettingsPanel.qml` (diagnostics), `AmbientOverlay.qml` (frameless overlay).
  System tray + global hotkeys + Core watchdog (max 4 relaunch attempts).
- **Build/install — `build/`** (`build_core.spec` / `build_gui.spec` /
  `build_hydrate.spec` + matching `.bat`) and `installer/ALISON_Setup.iss`.

## 3. Key Design Decisions (v3 — keep)

1. **Screenpipe is optional and flag-gated** (default OFF, not bundled).
2. **`capture_policy.json` is the canonical exclusion source** consumed by both the
   perception and action layers.
3. **`action_policy.json` owns the security model**: tier mapping, `execute_cmd`
   sandbox (confirmation + regex allowlist + workdir jail), kill switch, per-app
   capability integration registry.
4. **Kill switch** (`ctrl+alt+k` / IPC `kill_switch`) disables the executor **and**
   terminates any in-flight `execute_cmd` subprocess.
5. **Tier-3 (privileged) is scaffold-only** — real privileged actions need explicit
   policy approval before use.
6. **First real session runs with `ALISON_GLOBAL_DRY_RUN` ON** (dry-run env hook in
   `action_policy.json:execute_cmd.global_dry_run_env`).
7. **Frozen builds route state/models to `%LOCALAPPDATA%\A.L.I.S.O.N.`** (install dir
   under Program Files is read-only for standard users).
8. **IPC uses loopback TCP + named shared memory** (no disk files).
9. **Core stdout/stderr → `%LOCALAPPDATA%\A.L.I.S.O.N.\logs\alison_core.out.log`**
   (UAC-safe; not next to the exe).

## 4. Environment & Toolchain

| Item | Path / version |
|---|---|
| Engine Python | `C:\Users\dgc12\AppData\Local\Programs\Python\Python310\python.exe` (3.10.11) |
| Engine deps | torch **2.13.0+cu126** (CUDA 12.6, `torch.cuda.is_available()==True`), llama_cpp **0.2.90**, PyInstaller 6.21.0 |
| GPU | NVIDIA GeForce RTX 2080 SUPER (8 GB VRAM); system 32 GB RAM |
| GUI venv | `A.L.I.S.O.N\gui\.venv` (Py 3.12; PyQt6, pyzmq, pywin32, keyboard, psutil, numpy) |
| Installer | `C:\Users\dgc12\AppData\Local\Programs\Inno Setup 6\ISCC.exe` |
| Model weights | `C:\Users\dgc12\AppData\Local\A.L.I.S.O.N\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.92 GB, present) |
| Runtime log | `C:\Users\dgc12\AppData\Local\A.L.I.S.O.N\logs\alison_core.out.log` |

**Run (dev):**
- Core: `python310 src\alison_core.py --ipc --auto` (from `A.L.I.S.O.N`). `--auto` runs N cycles (`--cycles`, default 10) then exits; `--ipc` enables the control/telemetry channels.
- GUI: `A.L.I.S.O.N\gui\.venv\Scripts\python.exe gui\alison_gui.py`. It auto-launches the Core after 1.5 s.

**Build:** `A.L.I.S.O.N\build\build_core.bat`, `build_gui.bat`, `build_hydrate.bat`.
**Installer:** `ISCC.exe A.L.I.S.O.N\installer\ALISON_Setup.iss` → `installer\installer_output\ALISON_Setup.exe`.
**Tests (engine):** `src\tests\test_{larimar_unbinding,screenpipe_bridge,action_security}.py` — all pass.

## 5. Operational Constraints & Gotchas (read before doing anything)

1. **Foreground shell commands are killed at ~900–920 s.** Anything longer MUST run
   detached (e.g. `Start-Process cmd /c <bat>` — expect a harmless `ChildProcess.kill`
   harness message; the process survives). Add a **lock file** at the top of the bat
   so the harness's ~2.5-min duplicate retry cannot start a second concurrent build.
   (Core PyInstaller build now takes ~16 min; ISCC ~13.5 min.)
2. **Dev vs frozen model path:** dev Core looks in `src\models\` (not present → Phase 1
   crashes with `ValueError: Model path does not exist`); frozen Core uses
   `%LOCALAPPDATA%\A.L.I.S.O.N.\models`. `ALISON_MODELS_DIR` env overrides everything.
3. **Core IPC starts only AFTER the 8B model loads** (`_ipc_init()` at
   `alison_core.py:3828`). `get_status` will not answer until ~2–4 min into boot.
4. **Redirected stdout is block-buffered** — the Core log can stay empty while the
   process is healthy. Use `PYTHONUNBUFFERED=1` or query IPC instead of trusting the log.
5. **`--auto --cycles N` self-terminates** after N cycles (N=4 ≈ ~5–7 min incl. boot);
   use it for bounded verification runs.
6. **IPC control REQ** has no server timeout by default — always set
   `zmq.RCVTIMEO` (GUI bridge already does; any new client must too) or it hangs.
7. **Two Cores can't share ports**: the second crashes at `_ipc_init` (bind conflict).
   Ensure only one instance before verification.
8. **Git:** repo root is the workspace (`E:\ClaudeATHome\Projects\Coding Projects`);
   files live under `A.L.I.S.O.N\`. Binaries, `*.pt`, `.venv`, models, logs, and
   `alison_genome.json`/`ica_persona.json` are gitignored. Don't commit those.
9. **Long-run artifact side-effects:** running the dev Core from the repo root writes
   `ica_brain.pt` / `ica_persona.json` / `alison_genome.json` into the CWD. Clean up
   or gitignore them.

## 6. IPC Contract

- **Telemetry:** `tcp://127.0.0.1:5557` — named shared memory + PUB/SUB topics
  (`token_stream`, `screen_context`, `thought`, `log`, `ear_state`, ...).
- **Control:** `tcp://127.0.0.1:5558` — REQ/REP JSON. Verbs:
  `get_status`, `set_screen_sense`, `toggle_screen_sense`, `set_gamma_bounds`,
  `set_action_executor`, `execute_action`, `kill_switch`, `user_speech`,
  `start_listen`, `set_wakeword`.
- **`get_status` reply (v3, enriched):**
  `ok, core_version, uptime_s, device, gpu_name, ram_mb, vram_mb, model_loaded,
  model_path, model_size_bytes, screen_sense_enabled, screenpipe_enabled,
  gamma_bounds, gamma, action_executor_enabled, cuda_paused`.

## 7. GUI / QML Bridge Contract

- **Properties:** `gamma`, `drives`, `affect`, `activity`, `status`, `coreOnline`,
  `listening`, `diagnostics` (map: `coreVersion, uptime_s, device, gpuName, ramGB,
  vramGB, modelLoaded, modelPath, modelFile, modelSizeGB, screenSense, screenpipe,
  actionExecutor, gamma, corePid, coreFailures, installDir, stateDir, modelsDir,
  logPath, logTail`).
- **Slots (QML-invokable):** `sendCommand(action, params)`, `launchCore()`,
  `stopCore()`, `refreshDiagnostics()`, `fetchStatus()`, `openLog()`,
  `toggleOverlay()`, `moveOverlay(dx,dy)`.
- **Signals:** `telemetryUpdated`, `eventReceived(topic,text)`, `overlayRequested`,
  `statusChanged`, `diagnosticsChanged`.
- **Refresh cadence:** diagnostics map auto-refreshes every 3 s (QTimer) via
  `refreshDiagnostics` + `fetchStatus`.
- **Global hotkeys:** `alt+space` overlay, `ctrl+alt+a` dashboard, `alt+v` listen,
  `ctrl+alt+k` kill switch.

## 8. Work Log (what has been done)

### v3 feature work
- **W0** — `src/config/capture_policy.json` (canonical perception/action exclusions).
- **W1** — Larimar unbinding: `forget_pattern()` + `consolidate()` in the hippocampal
  memory index; `test_larimar_unbinding.py` passes (M_t 1314.9→960.1, recall degraded,
  sibling intact).
- **W2** — `src/alison_screenpipe.py` optional ingest bridge; `test_screenpipe_bridge.py`
  passes (transition-gated writes, dedupe, exclusion, offline).
- **W3** — `src/config/action_policy.json` + 3-tier action security in
  `alison_actions.py`, wired into Core (dispatch_action, IPC verbs, proactive-loop
  hooks) + GUI `ctrl+alt+k`; `test_action_security.py` passes (tiers, allowlist/
  confirmation gating, kill switch blocks AND terminates in-flight subprocess).
  Bugs caught/fixed during W3: tier-fallthrough default, `workdir_jail` makedirs.
- **W4/W5** — `docs/architecture_v3.md`, README v3 sections, `build/verify_paths.py` 100%.

### Fixes shipped (commit log, `main`)
| Commit | What |
|---|---|
| `e6eb091` | initial self-contained engine + GUI + build + installer |
| `d9019e7` | workspace README + skip interactive post-install when silent |
| `a79461b` | **v3** (Screenpipe adapter, Larimar unbinding, action security) |
| `dff209a` | installer VRAM detection across ALL video controllers (was reading only index 0) + non-blocking failure |
| `ef092e9` | bundle v3 policy configs in frozen Core; cap watchdog to 4 relaunches; terminate Core on GUI exit |
| `bf70fae` | UAC-safe Core log path (`%LOCALAPPDATA%`); `app_state_dir` makedirs |
| `e33ca19` | **root cause of "Core wouldn't load"**: fix PyInstaller TOC dest path for `llama_cpp\lib` (was treating a directory as a file → `Failed to extract llama_cpp\lib`); `upx=False` for build speed |
| `a2d67cf` | enrich `get_status` with device/model/hardware/uptime/screenpipe/version |
| `51e5813` | **GUI settings tab**: TabBar layout, `SettingsPanel.qml`, bridge diagnostics + Launch/Stop/Kill/Refresh, log tail, 3 s auto-refresh, watchdog hold-off on Stop, IPC send timeout |

## 9. Verification Status (verified this session)

- ✅ Frozen Core **boots end-to-end**: clean extraction → 8B model load → `AlisonIPC
  online` → cognition loop running (it answered `get_status` over IPC).
- ✅ `get_status` returns correct enriched fields: `device=cuda`,
  `gpu_name=NVIDIA GeForce RTX 2080 SUPER`, `ram_mb=32682`, `vram_mb=8191`,
  `core_version=3.0`, `action_executor_enabled=true`, `screenpipe_enabled=false`.
- ✅ QML (Dashboard + Settings) loads with zero QML errors against the Bridge.
- ✅ Tests pass: Larimar unbinding, Screenpipe bridge, Action security.
- ✅ Fresh binaries built: `dist\ALISON_Core.exe` (2,745 MB, 13:38),
  `dist\ALISON_GUI.exe` (76 MB, 13:40), `dist\ALISON_Hydrate.exe` (28 MB),
  `installer\installer_output\ALISON_Setup.exe` (2,795 MB, 14:04).

## 10. Current State & Open Items

- **User has not yet relaunched the new build** after the llama_cpp fix + settings tab;
  the primary next step is: reinstall `installer\installer_output\ALISON_Setup.exe`
  (or launch `dist\ALISON_GUI.exe`), wait ~2–4 min for Core boot, then open the
  **Settings** tab and confirm live status, paths, model, log tail, and the
  Launch/Stop/Kill controls.
- **Tier-3 (privileged) is scaffold-only** — needs a real privileged-action policy.
- **Screenpipe adapter is delivered but not wired into the runtime loop** (flag-gated).
- Possible future work: push `get_status` instead of polling, add GPU/CPU utilization
  telemetry, watchdog attempt-count exposure in the tray, installer auto-update via
  bundled `appcast.xml`.

## 11. Quick File Reference

| File | Role |
|---|---|
| `A.L.I.S.O.N/src/alison_core.py` | engine + IPC control handler (`_ipc_control_handler`), boot, consciousness loop |
| `A.L.I.S.O.N/src/alison_actions.py` | 3-tier action security, execute_cmd sandbox, kill switch |
| `A.L.I.S.O.N/src/alison_screenpipe.py` | optional Screenpipe ingest adapter (W2) |
| `A.L.I.S.O.N/src/alison_sense.py` | screen perception daemon |
| `A.L.I.S.O.N/src/alison_ipc.py` | shared-memory telemetry + ZMQ events/control |
| `A.L.I.S.O.N/src/config/{capture_policy,action_policy}.json` | v3 policies (bundled into frozen Core) |
| `A.L.I.S.O.N/gui/alison_gui.py` | Bridge, threads, tray, hotkeys, watchdog, diagnostics |
| `A.L.I.S.O.N/gui/qml/main.qml` | TabBar + StackLayout (Dashboard/Settings) |
| `A.L.I.S.O.N/gui/qml/SettingsPanel.qml` | diagnostics tab (status, paths, runtime, log tail) |
| `A.L.I.S.O.N/gui/theme.py` | HUD palette / constants |
| `A.L.I.S.O.N/build/build_{core,gui,hydrate}.spec` + `.bat` | PyInstaller pipeline |
| `A.L.I.S.O.N/installer/ALISON_Setup.iss` | Inno Setup installer (VRAM detection, hydrate post-install) |
| `A.L.I.S.O.N/docs/architecture_v3.md` | v3 architecture doc |
| `A.L.I.S.O.N/src/tests/` | engine tests |

## 12. Session 2 Work Log -- Hardening & Cognitive Synthesis (Phases 1-4)

Executed per the architectural spec ("Architectural Optimization, Operational
Hardening, and Cognitive Synthesis"): early IPC bootstrap, Win32 Job Object
containment, PerceptionGateway precision-weighted gating, and NVML hardware
telemetry + Tier 3 signing scaffold.

### Phase 1 -- Early IPC bootstrap (kills the boot blackout)
- `_ipc_init()` now runs at the **top of `__main__`** (before Phase 0) instead of
  after the 8B model load, so `get_status` answers and `boot_state` events flow
  during the entire 2-4 min boot.
- New boot-state machine in `alison_core.py`: `BOOTING_CALIBRATING` ->
  `BOOTING_POPULATING_FISHER` -> `BOOTING_LOADING_MODEL` -> `ONLINE_RUNNING`,
  driven by `_set_boot_phase()`; `get_status` returns `boot_phase`.
- GUI: bridge `RCVTIMEO` 1500 -> 3000 ms + `LINGER=0` on REQ sockets;
  SettingsPanel shows live "Boot phase"; EventThread handles `boot_state`.
- Doc-spec corrections applied: `alison_ipc` has no `recv_control/reply_control`
  (real API is `start_control(handler)` -- already a poller thread); the control
  handler was already boot-safe because `neocortex`, `active_inference`,
  `action_executor` are module-level.

### Phase 2 -- Win32 Job Object containment + unbuffered logs
- `ActionExecutor` creates a job with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`;
  every `execute_cmd` child spawns `CREATE_SUSPENDED`, is assigned to the job,
  then resumed via `ntdll.NtResumeProcess` (no psutil dep in the frozen bundle).
  Kill switch and timeout now terminate the **entire process tree** via
  `TerminateJobObject`; if the Core process dies, the kernel closes the job
  handle and kills all action children (no orphans).
- pywin32 API note: this Python310's `win32job` exposes `CreateJobObject` (not
  `CreateJobObjectW`) -- fallback added.
- Top of `alison_core.py`: `PYTHONUNBUFFERED=1` + `sys.stdout/stderr
  .reconfigure(line_buffering=True)`; GUI launchCore passes `PYTHONUNBUFFERED=1`.
- `Neocortex` pins `use_mmap=True, use_mlock=False`.
- New test: `src/tests/test_job_containment.py` (assignment, tree kill, signing
  gate) -- verified green.

### Phase 3 -- PerceptionGateway (precision-weighted perceptual gating)
- `PerceptionGateway` in `alison_sense.py`: rolling EMA generative prior over
  the 6-dim PC state; salience = `gamma * ||pc_t - prior||`; adaptive threshold
  (running mean + 1.5 sigma); prior always tracks reality (gate controls
  broadcast, not learning). First state and warmup pass unconditionally.
- Wired into `update_self_model_deterministic` (every-10-cycles affect pass):
  expected context no longer perturbs affect; novel context passes through.
- `get_status` exposes `screen_novelty` / `screen_gated`.
- Screenpipe feeds the gateway **only** when `ALISON_SCREENPIPE=1`
  (`_screenpipe_gateway_loop`, publishes `novelty` events); default OFF per v3.

### Phase 4 -- NVML hardware telemetry + Tier 3 signing scaffold
- `src/alison_hw.py`: NVML via `ctypes.WinDLL("nvml.dll")` (ships with every
  NVIDIA driver -- zero new pip deps, no spec changes). 1 Hz `hardware` events
  (`gpu_util_pct`, `vram_used_mb`, `vram_total_mb`, `gpu_temp_c`) + `read_once()`
  for `get_status`. Verified live on the RTX 2080 SUPER.
- GUI EventThread stashes `hardware` events into diagnostics; SettingsPanel
  shows live GPU line; no polling.
- `src/alison_signing.py`: HMAC-SHA256 policy signing scaffold; `action_policy
  .json` gains a `signing` block (`enabled:false`); when enabled, privileged
  actions are denied without a valid signature (key via `ALISON_POLICY_KEY`).

### Verification (this session)
- Job tree kill proven (cmd -> cmd -> PING all terminated; job handle close on
  process death also kills the tree).
- NVML read: `{gpu_util_pct: 34, vram_used_mb: 7038, vram_total_mb: 8192,
  gpu_temp_c: 62}`.
- `get_status` (direct handler): `boot_phase`, `device=cuda`, hw + novelty
  fields all present.
- QML smoke: `main.qml` loads with zero errors incl. new SettingsPanel rows.
- Tests green: Action security, Screenpipe bridge, Larimar unbinding, Job
  containment (new).
- Fresh binaries: `dist\ALISON_Core.exe`, `dist\ALISON_GUI.exe`,
  `installer\installer_output\ALISON_Setup.exe` (built via detached chain +
  lock files; see `C:\Users\dgc12\AppData\Local\Temp\opencode\run_all_builds.bat`).

### Remaining open items
- Relaunch the rebuilt GUI, open Settings, confirm live boot phase + GPU line.
- Tier 3: real privileged-action policy + key management (scaffold only).
- Screenpipe: still OFF by default (set `ALISON_SCREENPIPE=1` to enable).