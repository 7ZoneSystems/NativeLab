@echo off
setlocal enabledelayedexpansion

echo.
echo ==============================================
echo    Native Lab Pro — PyInstaller Build
echo                 Windows
echo ==============================================
echo.

REM ── Check dependencies ─────────────────────────

echo [INFO] Checking dependencies...

python -m PyInstaller --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] PyInstaller not found. Installing...
    python -m pip install pyinstaller
)

REM Check PyQt6
python -c "import PyQt6" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyQt6 not found.
    echo         Install with: pip install PyQt6
    exit /b 1
)

REM Check llama binaries
IF NOT EXIST "llama\bin\llama-cli.exe" (
    echo [ERROR] llama\bin\llama-cli.exe not found.
    echo         Download Windows build from:
    echo         https://github.com/ggerganov/llama.cpp/releases
    echo         Look for: llama-*-bin-win-x64.zip
    echo         Extract into llama\bin\
    exit /b 1
)

REM ── Clean previous build ───────────────────────

echo [INFO] Cleaning previous build...

IF EXIST dist\NativeLabPro rmdir /s /q dist\NativeLabPro
IF EXIST build rmdir /s /q build

REM ── Run PyInstaller ────────────────────────────

echo [INFO] Building with PyInstaller...

python -m PyInstaller NativeLab.spec --noconfirm --clean

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyInstaller build failed.
    exit /b 1
)

REM ── Post-build folders ─────────────────────────

echo [INFO] Creating required data folders...

mkdir dist\NativeLabPro\localllm 2>nul
mkdir dist\NativeLabPro\sessions 2>nul
mkdir dist\NativeLabPro\chat_refs 2>nul
mkdir dist\NativeLabPro\ref_cache 2>nul
mkdir dist\NativeLabPro\ref_index 2>nul
mkdir dist\NativeLabPro\paused_jobs 2>nul

REM ── Create launcher ────────────────────────────

echo [INFO] Creating launcher...

(
echo @echo off
echo cd /d "%%~dp0"
echo NativeLabPro.exe %%*
) > dist\NativeLabPro\run.bat

REM ── Done ───────────────────────────────────────

echo.
echo ==============================================
echo  Build complete!
echo  Output: dist\NativeLabPro\
echo.
echo  Next steps:
echo    1. Copy your .gguf model into:
echo       dist\NativeLabPro\localllm\
echo.
echo    2. Run the app:
echo       dist\NativeLabPro\run.bat
echo.
echo    3. Or double-click:
echo       dist\NativeLabPro\NativeLabPro.exe
echo ==============================================
echo.

pause