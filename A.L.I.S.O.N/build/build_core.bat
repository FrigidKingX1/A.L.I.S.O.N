@echo off
setlocal
:: ============================================================================
:: A.L.I.S.O.N. -- CORE binary build (PyInstaller)
:: ----------------------------------------------------------------------------
:: Run from A.L.I.S.O.N/build. Uses the engine Python (the one that already
:: runs alison_core.py with torch + llama_cpp + PyInstaller installed).
:: Output: ..\dist\ALISON_Core.exe  (console)
:: ============================================================================
set PYTHON=C:\Users\dgc12\AppData\Local\Programs\Python\Python310\python.exe
set HERE=%~dp0

if not exist "%PYTHON%" (
    echo [build_core] Engine Python not found at %PYTHON%
    echo [build_core] Set PYTHON= to the interpreter that runs alison_core.py
    exit /b 1
)

echo [A.L.I.S.O.N.] Building ALISON_Core.exe (PyInstaller)...
"%PYTHON%" -m PyInstaller "%HERE%build_core.spec" ^
    --noconfirm --clean ^
    --distpath "%HERE%..\dist" ^
    --workpath "%HERE%work"

if errorlevel 1 (
    echo [build_core] FAILED
    exit /b 1
)

echo [build_core] OK: ..\dist\ALISON_Core.exe
endlocal
