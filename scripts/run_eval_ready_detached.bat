@echo off
cd /d "C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev"
"C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev\.venv\Scripts\python.exe" -u scripts\eval_all_detectors.py --only FireSVMDetector HOGSVMDetector MobileNetSSDDetector MVSTGCNDetector --out reports\full_corpus_eval_ready.json > "logs\eval_ready_detached.log" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "logs\eval_ready_detached.log"
