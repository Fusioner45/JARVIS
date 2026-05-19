@echo off
setlocal enabledelayedexpansion
title JARVIS Unified Setup (V10)

echo.
echo  ================================================
echo        JARVIS UNIFIED PRODUCTION SETUP
echo  ================================================
echo.

:: 1. Python Check
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    pause
    exit /b
)

:: 2. Venv Management
if not exist ".venv" (
    echo [JARVIS] Creation de l'environnement virtuel ^(.venv^)...
    python -m venv .venv
)

echo [JARVIS] Activation de l'environnement virtuel...
set "VENV_PYTHON=.venv\Scripts\python.exe"
set "VENV_PIP=.venv\Scripts\pip.exe"

if not exist "!VENV_PYTHON!" (
    echo [ERREUR] Impossible de trouver l'interpreteur dans .venv.
    pause
    exit /b
)

:: 3. Core Updates
echo [JARVIS] Mise a jour de pip...
!VENV_PYTHON! -m pip install --upgrade pip --quiet

:: 4. Hardware Detection
echo [JARVIS] Detection du materiel...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Aucun GPU NVIDIA detecte. Installation mode CPU.
    set INSTALL_MODE=CPU
) else (
    echo [OK] GPU NVIDIA detecte. Installation mode GPU ^(CUDA 12.1^).
    set INSTALL_MODE=GPU
)

:: 5. Dependency Installation
echo [JARVIS] Installation des dependances Core (requirements.txt)...
!VENV_PYTHON! -m pip install -r requirements.txt

if "!INSTALL_MODE!"=="GPU" (
    echo [JARVIS] Installation de la stack AI ^(CUDA 12.1^)...
    !VENV_PYTHON! -m pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu121
) else (
    echo [JARVIS] Installation de la stack AI ^(CPU^)...
    !VENV_PYTHON! -m pip install torch==2.2.1+cpu torchvision==0.17.1+cpu torchaudio==2.2.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
)

:: 6. Model Downloader Integration
echo [JARVIS] Preparation des modeles...
if not exist "models" mkdir models
if not exist "models\silero_vad.onnx" (
    echo [JARVIS] Telechargement de Silero VAD ^(ONNX^)...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx' -OutFile 'models/silero_vad.onnx'"
)

if not exist "models\kokoro-v1.0.onnx" (
    echo [JARVIS] Telechargement de Kokoro ONNX...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx' -OutFile 'models/kokoro-v1.0.onnx'"
)

if not exist "models\voices-v1.0.bin" (
    echo [JARVIS] Telechargement des voix Kokoro...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin' -OutFile 'models/voices-v1.0.bin'"
)

:: 7. Final Validation & Self-Repair
echo [JARVIS] Validation finale de l'installation...
!VENV_PYTHON! -c "import PyQt6; print('PyQt6 OK'); import torch; print('Torch OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Verification echouee. Tentative de reparation PyQt6...
    !VENV_PYTHON! -m pip install --force-reinstall PyQt6 PyQt6-Qt6
)

!VENV_PYTHON! -m pip check

echo.
echo ================================================
echo [DONE] JARVIS est pret !
echo Mode : !INSTALL_MODE!
echo.
echo [IMPORTANT] Le premier lancement telechargera les modeles Whisper.
echo Cela peut prendre plusieurs minutes.
echo.
echo Pour lancer JARVIS :
echo .venv\Scripts\python.exe main.py
echo ================================================
echo.

pause
endlocal
