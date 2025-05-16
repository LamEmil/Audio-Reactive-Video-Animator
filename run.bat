@echo off
REM This batch file launches the Audio Reactive Video Animator application.

REM Get the directory where this batch file is located.
SET SCRIPT_DIR=%~dp0

REM Change the current directory to the script's directory.
REM This ensures that Python can find all the module files (main_gui.py, worker.py, etc.)
cd /d "%SCRIPT_DIR%"

REM Echo the current directory for debugging (optional)
REM echo Running from: %CD%

REM Check if main_gui.py exists in the current directory
IF NOT EXIST "main_gui.py" (
    echo ERROR: main_gui.py not found in the current directory:
    echo %CD%
    echo Please ensure run.bat is in the same directory as main_gui.py.
    pause
    exit /b 1
)

REM Launch the Python application.
REM Ensure 'python' is in your system PATH.
REM If you use a specific python environment (e.g., venv),
REM you would activate it before this line or call the python executable directly from the venv.
echo Launching Audio Reactive Video Animator...
python main_gui.py

REM Pause at the end to see any output/errors if the Python script exits quickly (optional)
REM If the Python script has its own GUI and runs until closed, this pause might not be necessary.
REM pause
