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
    # v3 policy/config consumed at runtime by alison_actions / alison_screenpipe.
    # Preserve the "config/" subdir so the modules (which resolve config/*.json
    # relative to their own __file__) find them inside the frozen bundle.
    (os.path.join(CONFIG, "action_policy.json"), "config"),
    (os.path.join(CONFIG, "capture_policy.json"), "config"),
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
        # collect_all() returns (src, dest_dir) 2-tuples for both datas and
        # binaries. The dest_dir is the package-relative DIRECTORY (e.g.
        # "llama_cpp\lib"), NOT the full file path. PyInstaller's TOC needs the
        # full destination name (dir + filename); otherwise it tries to extract
        # a FILE named "llama_cpp\lib" and fails with
        # "Failed to extract llama_cpp\lib: failed to open target file", which
        # aborts the frozen Core at bootstrap. Reconstruct the full dest, route
        # shared libs (.dll/.pyd/.so) to binaries, dedupe by dest.
        _seen = set()
        for entry in list(d) + list(b):
            if len(entry) == 2:
                src, dest_dir = entry
            else:
                dest_dir, src = entry[0], entry[1]
            if os.path.isdir(src) or src.endswith((".dist-info", ".egg-info", ".egg-link")):
                continue
            dest = os.path.join(dest_dir, os.path.basename(src)) if len(entry) == 2 else dest_dir
            if dest in _seen:
                continue
            _seen.add(dest)
            if src.lower().endswith((".dll", ".pyd", ".so", ".dylib")):
                a.binaries.append((dest, src, "BINARY"))
            else:
                a.datas.append((dest, src, "DATA"))
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
    upx=False,
    runtime_tmpdir=None,
    console=True,
    icon=ICON,
)
