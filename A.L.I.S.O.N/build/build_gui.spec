# -*- mode: python ; coding: utf-8 -*-
# ============================================================================
# A.L.I.S.O.N. -- GUI PyInstaller spec (lives in A.L.I.S.O.N/build).
# GUI source is at ../gui; engine bridge (alison_ipc) at ../src.
# Produces ALISON_GUI.exe (windowed) into the shared ../dist payload dir.
# ============================================================================
import os

HERE = SPECPATH
GUI = os.path.join(HERE, "..", "gui")
SRC = os.path.join(HERE, "..", "src")
ICON = os.path.join(HERE, "..", "installer", "assets", "alison_icon.ico")

added_files = [
    (os.path.join(GUI, "qml"), "qml"),
    (os.path.join(GUI, "shaders"), "shaders"),
    (os.path.join(SRC, "alison_ipc.py"), "."),
    (os.path.join(GUI, "theme.py"), "theme.py"),
]

a = Analysis(
    [os.path.join(GUI, "alison_gui.py")],
    pathex=[GUI, SRC, HERE],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtWidgets",
        "PyQt6.QtGui",
        "PyQt6.QtQuick",
        "PyQt6.QtNetwork",
        "alison_ipc",
    ],
    excludes=["*.gguf", "matplotlib", "tkinter", "scipy", "pandas", "PIL", "IPython"],
    noarchive=False,
)

# PyQt6 plugins (QtQuick / QML / Shadertools) are collected automatically by
# PyInstaller's per-submodule hooks once the modules are hidden-imported above.

a.datas = [x for x in a.datas if not (x[0].endswith(".gguf"))]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ALISON_GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    icon=ICON,
    disable_windowed_traceback=False,
    target_arch=None,
)
