@echo off
setlocal
:: ============================================================================
:: A.L.I.S.O.N. -- GUI binary build (PyInstaller)
:: ----------------------------------------------------------------------------
:: Run from A.L.I.S.O.N/build. Uses the GUI venv (PyQt6, pyzmq, numpy, psutil,
:: keyboard, pywin32, PyInstaller) at ..\gui\.venv.
:: Output: ..\dist\ALISON_GUI.exe  (windowed)
:: ============================================================================
set PYTHON=%~dp0..\gui\.venv\Scripts\python.exe
set HERE=%~dp0

if not exist "%PYTHON%" (
    echo [build_gui] GUI venv not found at %PYTHON%
    echo [build_gui] Create it: py -3.10 -m venv ..\gui\.venv
    echo [build_gui] then: ..\gui\.venv\Scripts\pip install -r ..\gui\requirements.txt
    exit /b 1
)

echo [A.L.I.S.O.N.] Building ALISON_GUI.exe (PyInstaller)...
"%PYTHON%" -m PyInstaller "%HERE%build_gui.spec" ^
    --noconfirm --clean ^
    --distpath "%HERE%..\dist" ^
    --workpath "%HERE%work"

if errorlevel 1 (
    echo [build_gui] FAILED
    exit /b 1
)

echo [build_gui] OK: ..\dist\ALISON_GUI.exe
endlocal
