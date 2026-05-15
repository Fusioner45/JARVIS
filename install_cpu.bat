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
python -m pip install --upgrade pip

echo [JARVIS] Installation des dependances Core (requirements.txt)...
pip install -r requirements.txt

echo.
echo [DONE] JARVIS est pret pour une utilisation CPU.
pause
endlocal
