@echo off
setlocal enabledelayedexpansion
title JARVIS UNIVERSAL INSTALLER

echo ============================================
echo   JARVIS - Installation Universelle
echo ============================================
echo.

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

echo.
echo [JARVIS] Etape 2 : Detection du GPU NVIDIA...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] GPU NVIDIA detecte !
    echo [JARVIS] Installation de PyTorch GPU (CUDA 12.1)...
    python -m pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu121
    set PYTORCH_TYPE=GPU
) else (
    echo [INFO] Aucun GPU NVIDIA detecte. Installation en mode CPU.
    echo [JARVIS] Installation de PyTorch CPU...
    python -m pip install torch==2.2.1+cpu torchvision==0.17.1+cpu torchaudio==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu
    set PYTORCH_TYPE=CPU
)

echo.
echo [JARVIS] Verification finale de l'installation...
python -c "import PyQt6; print('PyQt6 OK'); import torch; print('Torch OK'); import aiohttp; print('aiohttp OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] L'installation semble incomplete. Verifiez les logs de pip ci-dessus.
    echo Tentative de reparation forcee...
    python -m pip install --force-reinstall PyQt6 PyQt6-Qt6
)

echo.
echo ============================================
echo   CONFIGURATION RECOMMANDEE
echo ============================================
echo.

REM Verifier FFmpeg
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] FFmpeg non detecte (optionnel mais recommande)
    echo    Commande : winget install ffmpeg
    echo.
) else (
    echo [OK] FFmpeg deja installe
    echo.
)

REM Verifier Ollama
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama non detecte (OBLIGATOIRE pour LLM)
    echo    Telecharger : https://ollama.ai
    echo.
) else (
    echo [OK] Ollama deja installe
    echo.
)

echo ============================================
echo   INSTALLATION COMPLETE !
echo ============================================
echo.
echo Configuration : %PYTORCH_TYPE%
echo.
echo [IMPORTANT] Avant de lancer main.py :
echo   1. Verifiez que Ollama tourne : ollama serve
echo   2. Copiez .env.example en .env : copy .env.example .env
echo   3. Editez .env avec vos parametres (HA_URL, tokens, etc)
echo   4. (Optionnel) Installez FFmpeg pour une meilleure TTS
echo.
echo Commande de demarrage : python main.py
echo.
pause
endlocal
