@echo off
setlocal

echo [JARVIS] GPU INSTALL (CUDA 12.1)

python -m pip install --upgrade pip

pip install -r requirements.txt

pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install onnxruntime-gpu==1.17.0

echo [DONE GPU]
pause
endlocal
