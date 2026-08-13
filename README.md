# A.L.I.S.O.N. — Workspace

A.L.I.S.O.N. (**A**dvanced **L**ogical **I**ntegrated **S**entient **O**perational **N**etwork)
is a self-contained desktop AGI: a local cognitive engine with a PyQt6 / QML ambient
dashboard, local neural voice (Kokoro + pyttsx3) and local speech recognition
(faster-whisper), packaged as standalone Windows executables and a one-click
Inno Setup installer.

## On-disk workspace layout (`Coding Projects`)

```
Coding Projects/
├── A.L.I.S.O.N/            ← tracked in git (this repository)
│   ├── src/                engine: alison_core, alison_voice, alison_ear,
│   │                       alison_ipc, alison_sense, alison_actions,
│   │                       alison_screenpipe (optional Screenpipe adapter),
│   │                       config/ (genome, persona, capture_policy,
│   │                       action_policy), tests/
│   ├── gui/                PyQt6 ambient dashboard (alison_gui.py, qml/,
│   │                       shaders/, requirements.txt, .venv)
│   ├── installer/          ALISON_Setup.iss + assets/ (icon, wizard logo,
│   │                       appcast.xml)
│   ├── build/              PyInstaller specs + build_*.bat + verify_paths.py
│   ├── dist/               built executables (gitignored):
│   │                       ALISON_Core.exe, ALISON_GUI.exe, ALISON_Hydrate.exe
│   ├── models/             downloaded model weights (gitignored)
│   └── logs/               runtime logs (gitignored)
├── Projects/               sibling projects (local only — NOT tracked)
├── Experiments/            research scratch (local only — NOT tracked)
└── Logs/                   aggregated logs (local only — NOT tracked)
```

## Build & package pipeline

1. **Engine** (Python 3.10 / torch CUDA): `build\build_core.bat` → `dist\ALISON_Core.exe`.
2. **Dashboard + hydrator** (GUI venv, Python 3.12): `build\build_gui.bat`, `build\build_hydrate.bat`.
3. **Installer**: `ISCC.exe installer\ALISON_Setup.iss` bundles `dist\*` + assets →
   `installer\installer_output\ALISON_Setup.exe`.

Run `build\verify_paths.py` to confirm every cross-reference resolves before a release build.

## Installation (end user)

Run `ALISON_Setup.exe`:

1. **Hardware pre-flight** — Pascal scripts verify physical host RAM (≥ 16 GB) and
   physical GPU VRAM (≥ 8 GB) via WMI / Kernel APIs.
2. **Binary staging** — installs the executables and runtime `.dll`s to `{autopf}\ALISON\`.
3. **Lazy weight hydration** — `ALISON_Hydrate.exe` downloads and SHA-256 validates
   `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (~4.92 GB) into
   `%LOCALAPPDATA%\A.L.I.S.O.N.\models`.
4. **OS integration** — registers startup persistence in
   `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`, launches `ALISON_GUI.exe`
   into the System Tray, and binds global hotkeys.

Runtime state for frozen builds resolves to `%LOCALAPPDATA%\A.L.I.S.O.N.`.

## v3 Capabilities

### Build vs. Adopt
A.L.I.S.O.N. deliberately **builds** its core cognition (ICA / Active Inference /
Larimar memory / local voice+STT) so the stack is auditable, offline-first, and
free of third-party runtime dependencies. For **perception grounding it selectively
adopts** best-in-class tooling behind a flag-gated, config-driven adapter — e.g.
Screenpipe for OCR/visual screen context — while keeping the home-grown
`alison_sense.py` (UIA window text + DeBERTa + Go-Emotions, fully offline) as the
default. Adopted components never become load-bearing: if the adapter is absent or
offline, the system degrades silently to its built perception layer.

See `A.L.I.S.O.N/docs/architecture_v3.md` for the full v3 design rationale.

### Optional Screenpipe Ingestion (flag-gated)
`src/alison_screenpipe.py` (`screenpipe_enabled`, default **OFF**) streams OCR /
visual frames from a running Screenpipe daemon into the same
`HippocampalMemoryIndex` used by the rest of A.L.I.S.O.N. It honors the canonical
exclusion list in `src/config/capture_policy.json` (sensitive apps / windows / URLs),
de-duplicates, gates writes on **attention-transition** events, and degrades to
offline mode if the daemon is unreachable. Screenpipe is **not** bundled in the
installer; the built-in `alison_sense.py` remains the default perception path.

### Action Security Model (§5.3)
`src/alison_actions.py` implements a **universal 3-tier capability dispatcher**
(Tier 1 = structured app integration, Tier 2 = UIA/Win32, Tier 3 = vision grounding
scaffold-only this release). All OS-mutating actions flow through one gate:

- **Tier gating** — `privileged` actions (e.g. `execute_cmd`, `delete_file`) require
  an allowlist match **or** explicit user confirmation; otherwise they are denied
  by default.
- **`execute_cmd` sandbox** — regex allowlist, a `workdir_jail` (auto-created),
  forced dry-run preview for any filesystem-mutating command, and explicit
  confirmation for everything else.
- **Kill switch** — `Ctrl+Alt+K` (or the `kill_switch` IPC command) sets
  `ActionExecutor.enabled = False` **and terminates any in-flight subprocess**
  (`Popen.terminate()`), halting all further OS interaction immediately.
- **Confirmation flow** — privileged actions are surfaced to the user and only
  executed when `confirmed=True` arrives over IPC.

Policy lives in `src/config/action_policy.json`; exclusions in
`src/config/capture_policy.json`. Verified by `src/tests/test_action_security.py`,
`test_larimar_unbinding.py`, and `test_screenpipe_bridge.py`.
