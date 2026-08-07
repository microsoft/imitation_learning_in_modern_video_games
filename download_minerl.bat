@echo off
REM Download MineRL (VPT) dataset from OpenAI's public blob storage
REM Usage:
REM   download_minerl.bat pixelbc/data/paper_data_split/minerl_6.13_treechop_train_files_10percent.txt
REM   download_minerl.bat pixelbc/data/paper_data_split/minerl_6.13_treechop_train_files.txt     (full dataset)

setlocal enableextensions enabledelayedexpansion

set SPLIT_FILE=%1

if "!SPLIT_FILE!"=="" (
    echo Usage: download_minerl.bat ^<split_file_path^>
    echo.
    echo Examples:
    echo   download_minerl.bat pixelbc\data\paper_data_split\minerl_6.13_treechop_train_files_10percent.txt
    echo   download_minerl.bat pixelbc\data\paper_data_split\minerl_6.13_treechop_train_files.txt
    exit /b 1
)

if not exist "!SPLIT_FILE!" (
    echo ERROR: split file not found: !SPLIT_FILE!
    exit /b 1
)

set OUTPUT_DIR=minerl_data
set BASE_URL=https://openaipublic.blob.core.windows.net/minecraft-rl/data/6.13/

if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"

echo ============================================================
echo Downloading MineRL data from OpenAI
echo Output dir: !OUTPUT_DIR!
echo Split file: !SPLIT_FILE!
echo ============================================================
echo.

setlocal enabledelayedexpansion
set /a count=0
for /f "usebackq delims=" %%A in ("!SPLIT_FILE!") do (
    set LINE=%%A
    REM Extract filename without extension (e.g., "demonstration_50259" from "demonstration_50259.mp4")
    for /f "tokens=1 delims=." %%B in ("!LINE!") do set NAME=%%B

    set /a count+=1
    echo [!count!] Downloading: !NAME!

    REM Download .mp4 file
    echo   - !NAME!.mp4
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('!BASE_URL!!NAME!.mp4', '!OUTPUT_DIR!\!NAME!.mp4')" 2>nul
    if !errorlevel! neq 0 (
        echo     WARNING: failed to download !NAME!.mp4
    )

    REM Download .jsonl file
    echo   - !NAME!.jsonl
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('!BASE_URL!!NAME!.jsonl', '!OUTPUT_DIR!\!NAME!.jsonl')" 2>nul
    if !errorlevel! neq 0 (
        echo     WARNING: failed to download !NAME!.jsonl
    )
)

echo.
echo ============================================================
echo Download complete! Data saved to: !OUTPUT_DIR!
echo ============================================================
echo.
echo Next step: train with
echo   train.bat minerl default.yaml data.data_path=./!OUTPUT_DIR!
