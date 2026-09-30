@echo off
title AgriScan AI - Crop Disease Detection Server
echo ============================================================
echo Starting AgriScan AI Web Application...
echo ABES Engineering College - HCLTech Industry Project
echo ============================================================
cd /d "%~dp0"

echo Opening browser at http://127.0.0.1:5000 ...
start http://127.0.0.1:5000

echo Launching Flask backend server...
.\venv\Scripts\python.exe app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server stopped with error code %ERRORLEVEL%.
    pause
)
