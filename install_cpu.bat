@echo off
setlocal

echo [JARVIS] CPU INSTALL

python -m pip install --upgrade pip
pip install -r requirements.txt

echo [DONE CPU]
pause
endlocal
