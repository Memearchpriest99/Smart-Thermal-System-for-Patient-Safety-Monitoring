@echo off
setlocal

echo ============================================================
echo  Ward Watcher -- Windows Build Script
echo ============================================================
echo.

REM Activate venv if present
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.8+ and retry.
    pause
    exit /b 1
)

REM Install / upgrade build dependencies
echo [1/3] Installing dependencies...
pip install flask flask-socketio pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

REM Clean previous build artefacts
echo [2/3] Cleaning previous build...
if exist build     rmdir /s /q build
if exist dist      rmdir /s /q dist

REM Run PyInstaller
echo [3/3] Building standalone package...
pyinstaller ward_watcher.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed. See output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Package location: dist\ward_watcher\
echo.
echo  To deploy:
echo    1. Copy the entire dist\ward_watcher\ folder to a USB stick.
echo    2. On the target PC, double-click ward_watcher.exe.
echo    3. A browser tab will open automatically at http://localhost:5000
echo ============================================================
echo.
pause
