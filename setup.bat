@echo off
REM Setup script: download datasets and pretrained models
REM Usage:
REM   setup.bat minerl              (download MineRL dataset)
REM   setup.bat csgo                (download CS:GO dataset - see README)

setlocal enableextensions enabledelayedexpansion

set TASK=%1

if "!TASK!"=="" (
    echo Usage: setup.bat ^<minerl^|csgo^>
    echo.
    echo minerl - Download VPT dataset for Minecraft tree-chopping task
    echo csgo   - Download CS:GO behavioral cloning dataset
    exit /b 1
)

call .venv\Scripts\activate.bat

if /i "!TASK!"=="minerl" (
    echo ============================================================
    echo Downloading MineRL (VPT) dataset...
    echo ============================================================
    echo.
    echo This downloads the VPT contractor data for the treechop task.
    echo Dataset path will be created as: minerl_data/
    echo.
    bash pixelbc/scripts/minerl_download_data.sh pixelbc/data/paper_data_split/minerl_6.13_treechop_train_files.txt
)

if /i "!TASK!"=="csgo" (
    echo ============================================================
    echo CS:GO Dataset Setup
    echo ============================================================
    echo.
    echo To download CS:GO data, follow the instructions at:
    echo   https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning#datasets
    echo.
    echo Then update your training config with:
    echo   data.data_path: ^<PATH_TO_CSGO_DATA^>
    exit /b 0
)

echo ERROR: unknown task '!TASK!'
exit /b 1
