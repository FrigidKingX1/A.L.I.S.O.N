@echo off
setlocal
:: ============================================================================
:: A.L.I.S.O.N. -- Hydrator binary build (PyInstaller)
:: ----------------------------------------------------------------------------
:: Run from A.L.I.S.O.N/build. Uses the GUI venv (huggingface_hub + PyInstaller)
:: at ..\gui\.venv. Freezes ..\src\hydrate.py into ALISON_Hydrate.exe.
:: Output: ..\dist\ALISON_Hydrate.exe  (console)
:: ============================================================================
set PYTHON=%~dp0..\gui\.venv\Scripts\python.exe
set HERE=%~dp0

if not exist "%PYTHON%" (
    echo [build_hydrate] GUI venv not found at %PYTHON%
    exit /b 1
)

echo [A.L.I.S.O.N.] Building ALISON_Hydrate.exe (PyInstaller)...
"%PYTHON%" -m PyInstaller "%HERE%build_hydrate.spec" ^
    --noconfirm --clean ^
    --distpath "%HERE%..\dist" ^
    --workpath "%HERE%work"

if errorlevel 1 (
    echo [build_hydrate] FAILED
    exit /b 1
)

echo [build_hydrate] OK: ..\dist\ALISON_Hydrate.exe
endlocal
