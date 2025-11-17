@echo off
echo ============================================================
echo Installing dependencies and running test...
echo ============================================================
echo.

echo [Step 1] Upgrading tokenizers and transformers...
python -m pip install --upgrade tokenizers transformers --quiet
if errorlevel 1 (
    echo [ERROR] Failed to upgrade packages
    pause
    exit /b 1
)

echo.
echo [Step 2] Installing other dependencies...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)

echo.
echo [Step 3] Installing package...
python -m pip install -e . --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install package
    pause
    exit /b 1
)

echo.
echo [Step 4] Running test...
echo.
python test.py

pause

