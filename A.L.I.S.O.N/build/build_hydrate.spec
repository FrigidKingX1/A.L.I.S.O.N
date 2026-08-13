# -*- mode: python ; coding: utf-8 -*-
# ============================================================================
# A.L.I.S.O.N. -- Hydrator PyInstaller spec (lives in A.L.I.S.O.N/build).
# Freezes ../src/hydrate.py (huggingface_hub) into ALISON_Hydrate.exe so the
# Inno Setup installer can fetch model weights on a machine with no Python.
# ============================================================================
import os

HERE = SPECPATH
SRC = os.path.join(HERE, "..", "src")
ICON = os.path.join(HERE, "..", "installer", "assets", "alison_icon.ico")

a = Analysis(
    [os.path.join(SRC, "hydrate.py")],
    pathex=[SRC, HERE],
    binaries=[],
    datas=[],
    hiddenimports=[
        "huggingface_hub",
        "huggingface_hub.hf_api",
        "huggingface_hub.file_download",
        "huggingface_hub.utils",
    ],
    excludes=[
        "torch", "tensorflow", "matplotlib", "tkinter",
        "scipy", "pandas", "PIL", "IPython", "numpy.testing",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ALISON_Hydrate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=True,
    icon=ICON,
    disable_windowed_traceback=False,
    target_arch=None,
)
