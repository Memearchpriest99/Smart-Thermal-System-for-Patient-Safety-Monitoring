@echo off
cd /d "C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev"
"C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev\.venv\Scripts\python.exe" -u scripts\train_contact_real_optuna.py --n-trials 30 --time-budget-hours 3 > "logs\train_contact_optuna.log" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "logs\train_contact_optuna.log"
