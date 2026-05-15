@echo off
setlocal
echo [JARVIS] Verification de l'environnement...

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
echo [JARVIS] Installation des dependances CPU CORE (requirements.txt)...
pip install -r requirements.txt

echo.
echo [JARVIS] Installation CPU terminee avec succes.
echo JARVIS peut maintenant demarrer en mode CPU uniquement.
pause
endlocal
