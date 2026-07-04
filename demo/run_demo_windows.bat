@echo off
rem Smart Thermal System demo — Windows replay launcher
cd /d "%~dp0.."
python -m demo.app --replay demo\sample_session
pause
