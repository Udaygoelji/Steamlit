@echo off
setlocal
IF NOT EXIST .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\prepare_dataset.py
python src\train.py --epochs 5
for /f %%f in ('dir /b /s data\animals\val\*.* ^| findstr /i ".jpg .png .jpeg"') do (
  set onefile=%%f
  goto :predict
)
:predict
python src\predict.py --image_path "%onefile%"
echo Done.
