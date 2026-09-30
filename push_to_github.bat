@echo off
title Push CarePulse AI to GitHub
echo ==============================================================================
echo   PUSHING HOSPITAL READMISSION ML SYSTEM TO GITHUB
echo   Target: https://github.com/ashenjaygit/Hospital_Readmission_ML_System.git
echo ==============================================================================
echo.
cd /d "%~dp0"

echo [1/2] Checking Git Branch & Remote...
git branch -M main
git remote -v
echo.

echo [2/2] Pushing to GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ==============================================================================
    echo [SUCCESS] Project pushed successfully to your GitHub!
    echo Visit: https://github.com/ashenjaygit/Hospital_Readmission_ML_System
    echo ==============================================================================
) else (
    echo.
    echo ==============================================================================
    echo [NOTICE] If you see 'Repository not found', please make sure you created
    echo the empty repository on GitHub first at: https://github.com/new
    echo Repository name: Hospital_Readmission_ML_System
    echo ==============================================================================
)

echo.
pause
