@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv-win\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found: .venv-win
    echo Follow the Windows setup steps in README.md first.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m lora_face_select_lite.gui
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] GUI exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
