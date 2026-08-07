@echo off
REM Download small CS:GO dataset (10 files, ~10-15 GB)
REM From: https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch

setlocal enableextensions enabledelayedexpansion

set OUTPUT_DIR=C:\csgo_data
set NUM_FILES=10
set LOG_FILE=csgo_download.log
set DOWNLOAD_URL=https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch/resolve/main/dataset_aim_expert

echo [%date% %time%] Starting CS:GO small dataset download >> "!LOG_FILE!"
echo.
echo ============================================================
echo CS:GO Small Dataset Download (10 files, ~10-15 GB)
echo ============================================================
echo Source: https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch
echo Output: !OUTPUT_DIR!
echo Log: !LOG_FILE!
echo.

REM Create output directory
if not exist "!OUTPUT_DIR!" (
    echo Creating directory: !OUTPUT_DIR!
    mkdir "!OUTPUT_DIR!"
)

echo Downloading %NUM_FILES% files from HuggingFace...
echo.

set /a SUCCESS=0
set /a FAILED=0

for /L %%i in (1,1,%NUM_FILES%) do (
    set FILE=hdf5_aim_july2021_expert_%%i.hdf5
    set /a PERCENT=(%%i * 100) / %NUM_FILES%

    echo [!PERCENT!%%] Downloading: !FILE! (%%i/%NUM_FILES%)

    REM Download from HuggingFace
    powershell -Command "try { (New-Object System.Net.WebClient).DownloadFile('!DOWNLOAD_URL!/!FILE!', '!OUTPUT_DIR!\!FILE!'); Write-Host 'OK' } catch { Write-Host 'FAILED: $_'; exit 1 }" 2>nul

    if !errorlevel! equ 0 (
        set /a SUCCESS+=1
        echo [OK] !FILE!
        echo [!PERCENT!%%] OK: !FILE! >> "!LOG_FILE!"
    ) else (
        set /a FAILED+=1
        echo [FAILED] !FILE!
        echo [!PERCENT!%%] FAILED: !FILE! >> "!LOG_FILE!"
    )
)

echo.
echo ============================================================
echo Download Summary
echo ============================================================
echo Successful: !SUCCESS!/%NUM_FILES!
echo Failed: !FAILED!/%NUM_FILES!
echo.

if !SUCCESS! gtr 0 (
    echo Downloaded files:
    dir /b "!OUTPUT_DIR!\*.hdf5" 2>nul
    echo.
    echo Ready to train!
    echo Next: csgo_complete.bat
) else (
    echo No files downloaded. Check internet connection and try again.
)

echo [%date% %time%] Download complete >> "!LOG_FILE!"
