@echo off
REM ============================================================
REM  One-click launcher for the Thermal Image Annotator.
REM  Prefers the standalone build (dist\ImageAnnotator.exe);
REM  falls back to running from source with Python.
REM ============================================================
cd /d "%~dp0"

if exist "dist\ImageAnnotator.exe" (
    start "" "dist\ImageAnnotator.exe"
    goto :eof
)

REM --- Fallback: run from source ---
where py >nul 2>nul
if %errorlevel%==0 (
    start "" py annotator.py
) else (
    start "" python annotator.py
)
