@echo off
echo [JARVIS] Installation des dependances Core...
pip install -r requirements.txt

echo [JARVIS] Installation de PyTorch (Version CPU Fallback)...
pip install torch torchvision torchaudio

echo [JARVIS] Installation terminee (CPU uniquement).
pause
