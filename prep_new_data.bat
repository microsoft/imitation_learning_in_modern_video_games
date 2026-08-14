@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat

REM Precompute DINOv2 RGB embeddings for the NEW files as they stream in, IN-PLACE
REM in C:\csgo_data_new. Safe to run repeatedly while download_csgo_stream.py is
REM still running -- it only lists COMPLETE .hdf5 (ignores .part) and writes
REM embeddings next to them. Do NOT move files out of csgo_data_new until the
REM download is fully done (the streamer would re-fetch anything missing).
REM
REM After the download completes, merge everything into the training dir:
REM   move C:\csgo_data_new\*.hdf5 C:\csgo_data\
REM   xcopy /E /I /Y C:\csgo_data_new\embeddings C:\csgo_data\embeddings
REM   then append the new names to pixelbc\data\paper_data_split\csgo_dm_train_files.txt

set "SRC=C:\csgo_data_new"
set "LIST=%TEMP%\csgo_new_files.txt"

if not exist "%SRC%" ( echo ERROR: %SRC% not found -- nothing downloaded yet. & exit /b 1 )

REM Filelist of COMPLETE .hdf5 only (.part partials are skipped by the glob).
dir /b "%SRC%\*.hdf5" > "%LIST%" 2>nul
for /f %%A in ('type "%LIST%" ^| find /c /v ""') do set "N=%%A"
if "%N%"=="0" ( echo No complete .hdf5 in %SRC% yet -- wait for the first file to land. & exit /b 0 )

echo Embedding %N% new file(s) from %SRC% (in-place, RGB DINOv2)...
python pixelbc\scripts\generate_embeddings.py --game csgo --encoder_config pixelbc\configs\csgo\aim_expert_dino_vitb.yaml --data_base_path "%SRC%" --filelist "%LIST%" --output_dir "%SRC%"

echo.
echo Done this pass. Re-run prep_new_data.bat as more files land.
endlocal
