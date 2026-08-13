# -*- coding: utf-8 -*-
"""Path-existence verifier for the A.L.I.S.O.N. restructured layout.
Run from A.L.I.S.O.N/build:  python verify_paths.py
Resolves every path referenced by the build specs / bats / installer relative
to its own anchor directory and reports PASS/FAIL without compiling anything.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # A.L.I.S.O.N
BUILD = os.path.join(ROOT, "build")
GUI = os.path.join(ROOT, "gui")
SRC = os.path.join(ROOT, "src")
INST = os.path.join(ROOT, "installer")
DIST = os.path.join(ROOT, "dist")
ASSETS = os.path.join(INST, "assets")

checks = []


def chk(name, path):
    ok = os.path.exists(path)
    checks.append((name, ok, path))
    return ok


# --- Anchor dirs ---
chk("A.L.I.S.O.N root", ROOT)
chk("build/", BUILD)
chk("gui/", GUI)
chk("src/", SRC)
chk("installer/", INST)
chk("dist/", DIST)
chk("installer/assets/", ASSETS)

# --- build specs resolve ../src + ../gui + ../installer/assets ---
SRC_CORE = os.path.join(SRC, "alison_core.py")
chk("build_core.spec -> ../src/alison_core.py", SRC_CORE)
for m in ("alison_ipc.py", "alison_sense.py", "alison_ear.py",
          "alison_voice.py", "alison_actions.py"):
    chk("src/" + m, os.path.join(SRC, m))
for c in ("alison_genome.json", "alison_self_model.json",
          "alison_persona.json", "ica_persona.json"):
    chk("src/config/" + c, os.path.join(SRC, "config", c))
chk("build_gui.spec -> ../gui/alison_gui.py", os.path.join(GUI, "alison_gui.py"))
chk("build_gui.spec -> ../gui/theme.py", os.path.join(GUI, "theme.py"))
chk("build_gui.spec -> ../gui/qml", os.path.join(GUI, "qml"))
chk("build_gui.spec -> ../gui/shaders", os.path.join(GUI, "shaders"))
chk("build_hydrate.spec -> ../src/hydrate.py", os.path.join(SRC, "hydrate.py"))
ICON = os.path.join(ASSETS, "alison_icon.ico")
chk("specs -> installer/assets/alison_icon.ico", ICON)

# --- bats resolve interpreters ---
PY310 = r"C:\Users\dgc12\AppData\Local\Programs\Python\Python310\python.exe"
chk("build_core.bat -> Python310 engine", PY310)
VENV_PY = os.path.join(GUI, ".venv", "Scripts", "python.exe")
chk("build_gui/build_hydrate.bat -> gui/.venv python", VENV_PY)

# --- installer references ---
chk("ALISON_Setup.iss -> ..\\dist", DIST)
for a in ("wizard_logo.bmp", "alison_icon.ico", "appcast.xml"):
    chk("ALISON_Setup.iss -> assets/" + a, os.path.join(ASSETS, a))
# payload exes expected in dist
for e in ("ALISON_GUI.exe", "ALISON_Hydrate.exe"):
    chk("dist/" + e + " (present or to-be-built)", os.path.join(DIST, e))
chk("dist/ALISON_Core.exe (to-be-built by build_core.bat)",
    os.path.join(DIST, "ALISON_Core.exe"))

# --- alison_gui.py ROOT resolution ---
gui_py = os.path.join(GUI, "alison_gui.py")
resolved_root = os.path.join(os.path.dirname(os.path.dirname(gui_py)), "src")
chk("alison_gui.py ROOT -> ../src", resolved_root)
chk("alison_gui.py ROOT contains alison_ipc.py",
    os.path.join(resolved_root, "alison_ipc.py"))

# --- Report ---
width = max(len(n) for n, _, _ in checks)
print("\nPATH-EXISTENCE VERIFIER")
print("=" * (width + 12))
fails = 0
for name, ok, path in checks:
    tag = "PASS" if ok else "FAIL"
    if not ok:
        fails += 1
    print(f"[{tag}] {name:<{width}}  {path}")
print("=" * (width + 12))
if fails:
    print(f"RESULT: {fails} FAILURE(S) -- fix before building.")
    sys.exit(1)
else:
    print("RESULT: ALL PATHS RESOLVE (100%). Safe to build.")
