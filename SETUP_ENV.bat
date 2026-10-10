@echo off
rem One-time setup: creates .venv inside this folder and installs requirements.txt into it.
rem Does not change the global Python installation.
setlocal
cd /d "%~dp0"
set "LOG=setup_log.txt"
echo Setup started %DATE% %TIME% > "%LOG%"

set "PYCMD="
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PYCMD (
    py -%%V -c "import struct,sys; sys.exit(0 if struct.calcsize('P')==8 else 1)" >nul 2>&1 && set "PYCMD=py -%%V"
  )
)
if not defined PYCMD (
  python -c "import struct,sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) and struct.calcsize('P')==8 else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
  echo.
  echo No compatible Python found. This project needs 64-bit Python 3.10, 3.11, 3.12 or 3.13
  echo ^(3.13 recommended; TensorFlow 2.20.0 has no Windows build for other versions^).
  echo Install Python 3.13 from https://www.python.org/downloads/ and run this file again.
  echo SETUP_FAILED no compatible Python >> "%LOG%"
  goto :fail
)

echo Using: %PYCMD%
%PYCMD% --version
%PYCMD% --version >> "%LOG%" 2>&1

if exist ".venv\Scripts\python.exe" (
  echo Virtual environment already exists, reusing it.
) else (
  echo Creating virtual environment in .venv ...
  %PYCMD% -m venv .venv >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo SETUP_FAILED venv creation >> "%LOG%"
    goto :fail
  )
)

echo Installing packages ^(about 1.5 GB, usually 5-20 minutes^). Details go to %LOG% ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 (
  echo SETUP_FAILED pip install >> "%LOG%"
  goto :fail
)

echo Checking the installation ...
".venv\Scripts\python.exe" -c "import tensorflow, sklearn, fastapi, streamlit; print('tensorflow', tensorflow.__version__, '| scikit-learn', sklearn.__version__, '| streamlit', streamlit.__version__)" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo SETUP_FAILED import check >> "%LOG%"
  goto :fail
)
echo SETUP_OK >> "%LOG%"
echo.
echo Setup finished. Start the project with RUN_PROJECT.bat
if /i not "%~1"=="nopause" pause
exit /b 0

:fail
echo.
echo Setup did not finish. Open %LOG% in this folder for the details.
if /i not "%~1"=="nopause" pause
exit /b 1
