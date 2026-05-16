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
python -m pip install -r requirements.txt

echo [JARVIS] Etape 2 : Installation de PyTorch CPU (OBLIGATOIRE)...
python -m pip install torch==2.2.1+cpu torchvision==0.17.1+cpu torchaudio==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu

echo.
echo [JARVIS] Verification finale de l'installation...
python -c "import PyQt6; print('PyQt6 OK'); import torch; print('Torch OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] L'installation semble incomplete. Verifiez les logs de pip ci-dessus.
    echo Tentative de reparation forcee de PyQt6...
    python -m pip install --force-reinstall PyQt6 PyQt6-Qt6
)

echo.
echo [INFO] Pour une latence TTS optimale, assurez-vous que FFMPEG est installe.
echo [INFO] Commande winget : winget install ffmpeg
echo.
echo [DONE] JARVIS est pret pour une utilisation CPU.
pause
endlocal
