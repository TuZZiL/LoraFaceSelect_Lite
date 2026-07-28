@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv-win\Scripts\pythonw.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found: .venv-win
    echo Follow the Windows setup steps in README.md first.
    pause
    exit /b 1
)

start "" "%PYTHON_EXE%" "%~dp0prepare_krea2_style_dataset_gui.py"
exit /b %ERRORLEVEL%
