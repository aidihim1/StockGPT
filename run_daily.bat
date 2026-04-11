@echo off
title StockGPT Daily Pipeline
cd /d "C:\Users\Dell\Desktop\stockGPT v3"

echo ============================================
echo  StockGPT Daily Pipeline
echo  %date% %time%
echo ============================================
echo.

echo [1/3] Updating NSE data...
python update_data.py
if %errorlevel% neq 0 (
    echo ERROR: update_data.py failed
    pause
    exit /b 1
)

echo.
echo [2/3] Running forecast...
python forecast.py
if %errorlevel% neq 0 (
    echo ERROR: forecast.py failed
    pause
    exit /b 1
)

echo.
echo [3/3] Opening dashboard...
start "" http://localhost:8501
streamlit run app.py --server.port 8501

pause
