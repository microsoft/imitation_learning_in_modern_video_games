@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Downloading all CS:GO datasets...
python download_all_datasets.py
pause
