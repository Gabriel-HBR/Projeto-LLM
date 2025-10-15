@echo off
chcp 65001 > nul
cls

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  MODO RAPIDO - SEM TREINAMENTO                           ║
echo ║  Detector de Toxicidade (Palavras-chave)                 ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo [INFO] Este modo usa classificacao baseada em palavras-chave
echo        Rapido e funciona em qualquer PC
echo.

REM Instalar dependências mínimas
echo Instalando dependencias minimas...
pip install pandas numpy openpyxl --quiet 2>nul

if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao instalar dependencias
    echo Execute manualmente: pip install pandas numpy openpyxl
    pause
    exit /b 1
)

echo [OK] Dependencias instaladas!
echo.
echo Iniciando aplicativo...
echo.

python app.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao executar o aplicativo
    echo Tente executar manualmente: python app.py
    pause
)



