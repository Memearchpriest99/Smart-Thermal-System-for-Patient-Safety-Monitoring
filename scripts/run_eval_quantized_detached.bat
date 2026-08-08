@echo off
cd /d "C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev"
"C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev\.venv\Scripts\python.exe" -u scripts\eval_quantized_models.py --out reports\full_corpus_eval_quantized.json > "logs\eval_quantized_detached.log" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "logs\eval_quantized_detached.log"
