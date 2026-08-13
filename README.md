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
│   │                       config/, tests/
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
