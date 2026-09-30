@echo off
title CarePulse AI - Inpatient Readmission Clinical Decision Support
echo ==============================================================================
echo   CAREPULSE AI: 30-DAY HOSPITAL READMISSION CLINICAL DECISION SUPPORT SYSTEM
echo   Healthcare Inpatient Risk Stratification & Discharge Planning
echo ==============================================================================
echo.
cd /d "%~dp0"

echo [1/2] Verifying Python Environment...
py --version >nul 2>&1
if %errorlevel% neq 0 (
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python environment not found!
        pause
        exit /b 1
    ) else (
        set PY_CMD=python
    )
) else (
    set PY_CMD=py
)

echo [2/2] Launching CarePulse Clinical Decision Support Dashboard...
echo Starting Streamlit server on http://localhost:8501
echo.
echo (Press Ctrl+C in this terminal anytime to stop the server)
echo ==============================================================================
echo.

%PY_CMD% -m streamlit run app.py --server.port 8501

pause
