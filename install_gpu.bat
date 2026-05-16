@echo off
setlocal
title JARVIS GPU INSTALLER (CUDA 12.1)

echo [JARVIS] Verification de l'environnement...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    pause
    exit /b
)

echo [JARVIS] Verification du materiel...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [ATTENTION] Aucun GPU NVIDIA detecte ou pilotes absents.
    echo L'installation va continuer mais pourrait echouer au runtime.
) else (
    echo [OK] GPU NVIDIA detecte.
)

echo.
echo [JARVIS] Mise a jour de pip...
python -m pip install --upgrade pip --quiet

echo [JARVIS] Etape 1 : Installation des dependances Core (requirements.txt)...
pip install -r requirements.txt

echo.
echo [JARVIS] Etape 2 : Installation de PyTorch GPU (CUDA 12.1)...
pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu121

echo.
echo [JARVIS] Validation finale...
pip check

echo.
echo [INFO] Pour une latence TTS optimale, assurez-vous que FFMPEG est installe.
echo [INFO] Commande winget : winget install ffmpeg
echo.
echo [DONE] JARVIS est pret avec acceleration RTX (CUDA 12.1).
pause
endlocal
