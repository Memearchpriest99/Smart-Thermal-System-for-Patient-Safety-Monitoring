@echo off
cd /d "C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev"
"C:\Users\guych\Desktop\Final Project\Smart-Thermal-System-for-Patient-Safety-Monitoring-dev\.venv\Scripts\python.exe" -u scripts\eval_onnx_models.py --out reports\full_corpus_eval_onnx.json > "logs\eval_onnx_detached.log" 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> "logs\eval_onnx_detached.log"
