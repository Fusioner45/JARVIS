@echo off
setlocal
echo [JARVIS] Verification de l'environnement GPU...

:: 1. Verification Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    pause
    exit /b
)

:: 2. Upgrade Pip
echo [JARVIS] Mise a jour de pip...
python -m pip install --upgrade pip

:: 3. Installation CPU CORE
echo [JARVIS] Installation des dependances CORE (requirements.txt)...
pip install -r requirements.txt

:: 4. Installation GPU EXTENSION (CUDA 12.1)
echo [JARVIS] Installation des accelerations GPU CUDA 12.1 (requirements-gpu.txt)...
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.17.0

echo.
echo [JARVIS] Installation GPU terminee avec succes (RTX 3070 Ti Ready).
pause
endlocal
