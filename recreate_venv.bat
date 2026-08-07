@echo off
setlocal enableextensions enabledelayedexpansion

echo Removing old venv...
rmdir .venv /s /q 2>nul

echo Creating fresh venv with Python 3.11...
uv venv --python 3.11

echo Activating venv...
call .venv\Scripts\activate.bat

echo Installing torch 2.7.0 with cu128 CUDA support (Blackwell sm_120)...
uv pip install --index-url https://download.pytorch.org/whl/cu128  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 
uv pip install -r requirements.txt

echo Installing pixelbc package...
uv pip install .

echo.
echo Verifying torch CUDA installation...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo Done! Venv ready at .venv\
pause
