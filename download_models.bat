@echo off
setlocal
title JARVIS Model Downloader

echo [JARVIS] Creation du dossier models...
if not exist "models" mkdir models

echo [JARVIS] Telechargement de Silero VAD (ONNX)...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx' -OutFile 'models/silero_vad.onnx'"

if %errorlevel% neq 0 (
    echo [ERREUR] Le telechargement a echoue. Verifiez votre connexion internet.
) else (
    echo [OK] Silero VAD telecharge avec succes dans models/silero_vad.onnx.
)

echo.
echo [DONE] Tous les modeles requis sont prets.
pause
endlocal
