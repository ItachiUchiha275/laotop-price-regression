@echo off
REM ============================================================
REM  CSE303 Laptop Price Regression Pipeline — Launcher
REM  Run: double-click or   .\run_pipeline.bat [scrape]
REM  Optional arg "scrape" re-scrapes Track B from the web.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  CSE303 Laptop Price Regression Pipeline
echo ============================================================

REM ---- 1. Check dependencies ------------------------------------
echo [1/4] Checking dependencies...
pip show pandas >nul 2>&1
if errorlevel 1 (
    echo   Installing requirements.txt...
    pip install -r requirements.txt
) else (
    echo   Dependencies OK.
)

REM ---- 2. Optional: Re-scrape Track B --------------------------
if /I "%1"=="scrape" (
    echo.
    echo [2/4] Scraping Track B (ryans.com + startech.com.bd)...
    python scripts/scrape_track_b.py
) else (
    echo.
    echo [2/4] Skipping scrape (pass "scrape" as argument to re-scrape).
)

REM ---- 3. Execute Track A notebook -----------------------------
echo.
echo [3/4] Running Track A (Kaggle) pipeline...
jupyter nbconvert --to notebook --execute notebooks/track_a_pipeline.ipynb --output track_a_output.ipynb --ExecutePreprocessor.timeout=300
if errorlevel 1 (
    echo   ERROR: Track A failed. Check track_a_output.ipynb
) else (
    echo   Track A complete.
)

REM ---- 4. Execute Track B notebook -----------------------------
echo.
echo [4/4] Running Track B (Scraped) pipeline...
jupyter nbconvert --to notebook --execute notebooks/track_b_pipeline.ipynb --output track_b_output.ipynb --ExecutePreprocessor.timeout=300
if errorlevel 1 (
    echo   ERROR: Track B failed. Check track_b_output.ipynb
) else (
    echo   Track B complete.
)

REM ---- Done ----------------------------------------------------
echo.
echo ============================================================
echo  Pipeline finished.
echo  Output notebooks: notebooks/track_a_output.ipynb
echo                     notebooks/track_b_output.ipynb
echo  Cleaned data:     data/track_a/cleaned/
echo                     data/track_b/cleaned/
echo  Report:           report/report_draft.md
echo ============================================================
pause
