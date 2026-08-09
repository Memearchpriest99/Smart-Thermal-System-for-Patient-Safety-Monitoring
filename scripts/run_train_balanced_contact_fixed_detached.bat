@echo off
cd /d "C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev"
"C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev\.venv\Scripts\python.exe" -u scripts\train_balanced_corpus.py --only-contact > "logs\train_balanced_contact_fixed_detached.log" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "logs\train_balanced_contact_fixed_detached.log"
