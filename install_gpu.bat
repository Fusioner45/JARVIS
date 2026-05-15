@echo off
echo [JARVIS] Installation des dependances Core...
pip install -r requirements.txt

echo [JARVIS] Installation de PyTorch CUDA 12.1 (RTX 3070 Ti)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [JARVIS] Installation terminee avec succes (GPU).
pause
