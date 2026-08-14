@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Re-download CS:GO deathmatch data from the PUBLIC HF dataset
REM TeaPearce/CounterStrike_Deathmatch (no HF_TOKEN needed). Each chunk = 200
REM files (~26 GB). Extracts .hdf5 into C:\csgo_data, skipping existing files.
REM
REM Usage:
REM   download_csgo_data.bat            -> first 1 chunk (files 1..200)
REM   download_csgo_data.bat 3          -> first 3 chunks (files 1..600)
REM   download_csgo_data.bat list       -> just list available chunks
REM   (for specific chunks: call download_csgo_data.py --tars <name>.tar ...)

set "ARG=%~1"
if "%ARG%"=="" set "ARG=1"

if /I "%ARG%"=="list" (
  python pixelbc\scripts\download_csgo_data.py --list
  goto :eof
)

echo.
echo Downloading %ARG% CS:GO chunk(s) from public HF -^> C:\csgo_data ...
echo.

python pixelbc\scripts\download_csgo_data.py --num-tars %ARG% --dest C:\csgo_data --stage C:\csgo_dl

echo.
echo Done. Add the new files to paper_data_split\csgo_dm_train_files.txt, then
echo re-run precompute_dino_embeddings.bat + precompute_dino_depth_embeddings.bat.
