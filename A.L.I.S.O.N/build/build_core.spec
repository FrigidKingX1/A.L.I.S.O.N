# -*- mode: python ; coding: utf-8 -*-
# ============================================================================
# A.L.I.S.O.N. -- CORE engine PyInstaller spec (lives in A.L.I.S.O.N/build).
# Engine source tree is at ../src; config JSON at ../src/config.
# Produces ALISON_Core.exe (console) into the shared ../dist payload dir.
# ============================================================================
import os

HERE = SPECPATH
SRC = os.path.join(HERE, "..", "src")
CONFIG = os.path.join(SRC, "config")
ICON = os.path.join(HERE, "..", "installer", "assets", "alison_icon.ico")

added_files = [
    (os.path.join(SRC, "alison_ipc.py"), "."),
    (os.path.join(SRC, "alison_sense.py"), "."),
    (os.path.join(SRC, "alison_ear.py"), "."),
    (os.path.join(SRC, "alison_voice.py"), "."),
    (os.path.join(SRC, "alison_actions.py"), "."),
    (os.path.join(CONFIG, "alison_genome.json"), "."),
    (os.path.join(CONFIG, "alison_self_model.json"), "."),
    (os.path.join(CONFIG, "alison_persona.json"), "."),
    (os.path.join(CONFIG, "ica_persona.json"), "."),
]

a = Analysis(
    [os.path.join(SRC, "alison_core.py")],
    pathex=[SRC, HERE],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        "torch", "llama_cpp",
        # Engine co-located modules.
        "alison_ipc", "alison_sense", "alison_ear", "alison_voice", "alison_actions",
        # Stage 2 voice + action intelligence.
        "faster_whisper", "kokoro", "sounddevice", "webrtcvad",
        "pyttsx3", "misaki", "espeakng_loader",
        "ctranslate2", "onnxruntime", "huggingface_hub", "tokenizers",
        "comtypes", "pywinauto", "pyautogui",
    ],
    excludes=["*.gguf", "matplotlib", "tkinter", "scipy", "pandas", "PIL", "IPython"],
    noarchive=False,
)

# Pull torch + llama_cpp fully (CUDA DLLs included automatically).
from PyInstaller.utils.hooks import collect_all
for pkg in ("torch", "llama_cpp"):
    try:
        d, b, h = collect_all(pkg)
        # collect_all() may return 2-tuples (src, dest); PyInstaller's TOC
        # needs 3-tuples (dest_name, src_name, typecode). Expand defensively,
        # and skip directories / package metadata (opening those as files on
        # Windows raises PermissionError).
        def _src_of(entry):
            if len(entry) == 2:
                return entry[0]
            return entry[1]
        for entry in d:
            src = _src_of(entry)
            if os.path.isdir(src) or src.endswith((".dist-info", ".egg-info", ".egg-link")):
                continue
            if len(entry) == 2:
                a.datas.append((entry[1], src, "DATA"))
            else:
                a.datas.append(tuple(entry))
        for entry in b:
            src = _src_of(entry)
            if os.path.isdir(src) or src.endswith((".dist-info", ".egg-info", ".egg-link")):
                continue
            if len(entry) == 2:
                a.binaries.append((entry[1], src, "BINARY"))
            else:
                a.binaries.append(tuple(entry))
        a.hiddenimports += h
    except Exception as exc:  # pragma: no cover
        print("collect_all warning for", pkg, ":", exc)

# Never bundle model weights.
a.datas = [x for x in a.datas if not x[0].endswith(".gguf")]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ALISON_Core",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=True,
    icon=ICON,
)
