@echo off
set PYTHONPATH=src
.\.venv\Scripts\python.exe -m uvicorn credit_risk.api:app --reload
