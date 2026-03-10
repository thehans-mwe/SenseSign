@echo off
:: SenseSpeak Server Launcher - auto-restarts on crash
:: Place in Startup folder or run manually

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
set "WEB_APP=%PROJECT_DIR%web_app.py"

echo ========================================
echo   SenseSpeak Server Launcher
echo ========================================

:loop
echo.
echo [%date% %time%] Starting SenseSpeak server...
cd /d "%PROJECT_DIR%"
"%VENV_PYTHON%" "%WEB_APP%"
echo.
echo [%date% %time%] Server stopped. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
