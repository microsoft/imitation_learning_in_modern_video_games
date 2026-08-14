@echo off
REM Resumable upload of the 329 USED CSGO training files (~45.7 GB) to horde.
REM Uses WSL rsync (3.2.7): survives dropped connections, skips already-uploaded
REM files, and excludes the unused\ subfolder. Prompts for the horde password once.
REM
REM Source : C:\csgo_data\*.hdf5  (the 329 files referenced by the train/val/test splits)
REM Dest   : horde (10.57.233.24) -> ~/csgo_data/
REM Rerun anytime -- it resumes where it left off.

echo Uploading used CSGO data (~45.7 GB) to horde via resumable rsync...
echo (excludes unused\ ; enter horde password when prompted)
echo.

wsl -d Ubuntu-24.04 rsync -avP --exclude=unused /mnt/c/csgo_data/ horde@10.57.233.24:~/csgo_data/

echo.
echo Done (or interrupted -- just rerun this .bat to resume).
