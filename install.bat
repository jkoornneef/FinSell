@echo off
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "VENV_PYTHON=%VENV%\Scripts\python.exe"
set "PYTHON_BOOTSTRAP="

pushd "%ROOT%" >nul || (
    echo Failed to access project directory: %ROOT%
    exit /b 1
)

if exist "%VENV_PYTHON%" goto install

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_BOOTSTRAP=py -3"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_BOOTSTRAP=python"
    )
)

if not defined PYTHON_BOOTSTRAP (
    echo Could not find a Python 3 interpreter.
    echo Install Python 3.10+ and ensure either 'py' or 'python' is available on PATH.
    popd >nul
    exit /b 1
)

echo Creating virtual environment in .venv ...
%PYTHON_BOOTSTRAP% -m venv "%VENV%"
if errorlevel 1 (
    echo Failed to create virtual environment.
    popd >nul
    exit /b 1
)

:install
echo Upgrading pip, setuptools, and wheel ...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo Failed to upgrade packaging tools.
    popd >nul
    exit /b 1
)

echo Installing finsell with dashboard and dev extras ...
"%VENV_PYTHON%" -m pip install -e ".[dashboard,dev]"
if errorlevel 1 (
    echo Failed to install project dependencies.
    popd >nul
    exit /b 1
)

echo.
echo Installation complete.
echo To activate the environment in Command Prompt:
echo   %VENV%\Scripts\activate.bat
echo To activate the environment in PowerShell:
echo   %VENV%\Scripts\Activate.ps1
echo.
echo To launch the dashboard:
echo   dashboard.bat

popd >nul
exit /b 0