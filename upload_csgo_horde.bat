@echo off
setlocal
REM Resumable upload of the 329 USED CSGO training files (~45.7 GB) to horde.
REM Uses WSL rsync (3.2.7): survives dropped connections, skips already-uploaded
REM files, and excludes the unused\ subfolder. Prompts for the horde password once.
REM
REM Source : C:\csgo_data\*.hdf5  (the 329 files referenced by the train/val/test splits)
REM Dest   : horde (default 10.86.82.17) -> /home/horde/csgo_data/
REM Rerun anytime -- it resumes where it left off.
REM
REM Arg 1: horde host/IP (default 10.86.82.17). e.g.  upload_csgo_horde.bat 10.57.233.24

set "HOST=%~1"
if not defined HOST set "HOST=10.86.82.17"
set "DEST=horde@%HOST%"
echo Target: %DEST%
echo.

echo Uploading used CSGO data (~45.7 GB) to horde via resumable rsync...
echo (excludes unused\ ; enter horde password when prompted)
echo.

wsl -d Ubuntu-24.04 rsync -avP --exclude=unused /mnt/c/csgo_data/ %DEST%:/home/horde/csgo_data/

echo.
echo Uploading cosmos_lora videos_stylized to horde...
wsl -d Ubuntu-24.04 rsync -avP --mkpath /mnt/c/workspace/world/flashdream_public/cosmos_lora/dataset/datasets/videos_stylized/ %DEST%:/home/horde/flashdream_public/cosmos_lora/datasets/videos_stylized/

echo.
echo Uploading cosmos_lora training code (scripts + cosmos-predict2.5, code only) to horde...
wsl -d Ubuntu-24.04 rsync -avP --mkpath --exclude=.git --exclude=__pycache__ --exclude=*.pth --exclude=*.safetensors --exclude=.venv --exclude=venv --exclude=checkpoints --exclude=dataset /mnt/c/workspace/world/flashdream_public/cosmos_lora/ %DEST%:/home/horde/flashdream_public/cosmos_lora/

echo.
echo Done (or interrupted -- just rerun this .bat to resume).
endlocal
