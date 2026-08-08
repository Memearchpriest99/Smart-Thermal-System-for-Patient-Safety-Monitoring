@echo off
cd /d "C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev"
"C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev\.venv\Scripts\python.exe" -u scripts\train_thermox3d_subset.py --stride 11 --time-budget-hours 11.5 --checkpoint-every 15 > "logs\train_thermox3d_subset_detached.log" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "logs\train_thermox3d_subset_detached.log"
