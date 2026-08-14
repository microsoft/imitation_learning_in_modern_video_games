@echo off
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

REM Download a NEW CS:GO deathmatch chunk into a SEPARATE directory
REM (C:\csgo_data_new) using the STREAMING HTTP path (Xet-free). The HF Xet
REM backend wedges when two downloads share its chunk cache; this streams the
REM tar over plain HTTP and unpacks each .hdf5 as it arrives -> ~26 GB written
REM (no 26 GB staging tar), real per-file progress, no deadlock. Skips any
REM .hdf5 already present.
REM
REM Usage:
REM   download_csgo_new.bat                                  -> default gap chunk 401_to_600, ALL 200 files
REM   download_csgo_new.bat hdf5_dm_july2021_601_to_800.tar  -> a specific chunk
REM   download_csgo_new.bat hdf5_dm_july2021_601_to_800.tar 50   -> stop after 50 files (~6.5 GB)
REM   download_csgo_new.bat list                             -> list available chunks, exit
REM
REM Fully-new (zero-overlap) chunks for the current dataset: 401_to_600,
REM 601_to_800, 801_to_1000, and everything 1201_to_1400 and up.

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" ( echo ERROR: .venv not found at %PYEXE% & exit /b 1 )

set "TAR=%~1"
if "%TAR%"=="" set "TAR=hdf5_dm_july2021_401_to_600.tar"
set "DEST=C:\csgo_data_new"

if /I "%TAR%"=="list" (
  "%PYEXE%" pixelbc\scripts\download_csgo_data.py --list
  goto :eof
)

REM Optional 2nd arg = max files (default 500).
set "MAXFILES=%~2"
if "%MAXFILES%"=="" set "MAXFILES=500"

echo.
echo ==========================================================
echo   Streaming %TAR%  (Xet-free HTTP)
echo   extract -^> %DEST%   max_files=%MAXFILES%
echo ==========================================================
echo.

"%PYEXE%" download_csgo_stream.py --tars %TAR% --dest "%DEST%" --max-files %MAXFILES%
if errorlevel 1 (
  echo.
  echo Download FAILED -- check disk space / network and retry.
  exit /b 1
)

echo.
echo Done. New .hdf5 are in %DEST%.
echo Next ^(manual^):
echo   1. Merge into the training data dir: move %DEST%\*.hdf5 C:\csgo_data\
echo   2. Add the new filenames to pixelbc\data\paper_data_split\csgo_dm_train_files.txt
echo   3. Re-run precompute_dino_embeddings.bat ^(+ _depth for RGB+D^), then retrain ^(early-stop ~epoch 4^).
endlocal
