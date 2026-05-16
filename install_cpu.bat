@echo off
setlocal
title JARVIS CPU INSTALLER

echo [JARVIS] Verification de l'environnement...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    pause
    exit /b
)

echo [JARVIS] Mise a jour de pip...
python -m pip install --upgrade pip --quiet

echo [JARVIS] Etape 1 : Installation des dependances Core (requirements.txt)...
pip install -r requirements.txt

echo [JARVIS] Etape 2 : Installation de PyTorch CPU (OBLIGATOIRE)...
pip install torch==2.2.1+cpu torchvision==0.17.1+cpu torchaudio==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu

echo.
echo [INFO] Pour une latence TTS optimale, assurez-vous que FFMPEG est installe.
echo [INFO] Commande winget : winget install ffmpeg
echo.
echo [DONE] JARVIS est pret pour une utilisation CPU.
pause
endlocal
