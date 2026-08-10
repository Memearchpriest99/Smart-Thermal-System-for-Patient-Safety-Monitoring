@echo off
cd /d "C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev"
"C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev\.venv\Scripts\python.exe" -u scripts\eval_config_d.py > "logs\eval_config_d.log" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "logs\eval_config_d.log"
