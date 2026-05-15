@echo off
setlocal
title JARVIS GPU INSTALLER (CUDA 12.1)

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
python -m pip install --upgrade pip

echo [JARVIS] ÉTAPE 1 : Installation des dependances Core...
pip install -r requirements.txt

echo.
echo [JARVIS] ÉTAPE 2 : Overlay des accelerations GPU (CUDA 12.1)...
:: Installation forcee pour remplacer la version CPU installee a l'etape 1
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.17.0

echo.
echo [DONE] JARVIS est pret avec acceleration RTX.
pause
endlocal
