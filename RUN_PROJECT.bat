@echo off
rem Starts the FastAPI prediction service and the Streamlit interface.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The virtual environment is missing. Run SETUP_ENV.bat first.
  pause
  exit /b 1
)
set "CAR_API_URL=http://127.0.0.1:8000"

echo Starting the prediction API on %CAR_API_URL% ...
start "AutoValue API" cmd /k ".venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1"

echo Waiting for the API to load the model ^(up to 3 minutes^) ...
set /a TRIES=0
:wait
curl -s -o nul -f %CAR_API_URL%/health && goto :up
set /a TRIES+=1
if %TRIES% geq 90 goto :apifail
timeout /t 2 /nobreak >nul
goto :wait

:apifail
echo The API did not start. Look at the "AutoValue API" window for the error.
pause
exit /b 1

:up
echo API is ready. Starting the Streamlit interface on http://localhost:8501 ...
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.headless true --server.port 8501
pause
