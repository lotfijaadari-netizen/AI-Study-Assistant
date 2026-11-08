@echo off
cd /d "%~dp0"

REM Libère le port 8501 si un ancien Streamlit est resté actif
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

call venv\Scripts\activate.bat
streamlit run app.py --server.port 8501