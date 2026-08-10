@echo off
cd /d "C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev"
"C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev\.venv\Scripts\python.exe" -u scripts\cv_contact_detectors.py > "logs\cv_contact.log" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "logs\cv_contact.log"
