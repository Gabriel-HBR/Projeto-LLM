@echo off
chcp 65001 > nul
cls
echo.
echo -----------------------------------------------------------------
echo    DETECTOR DE TOXICIDADE EM MENSAGENS
echo    Projeto com Transfer Learning
echo -----------------------------------------------------------------
echo.
echo Verificando dependencias...
pip install pandas numpy openpyxl --quiet 2>nul

if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao instalar dependencias
    echo Execute manualmente: pip install pandas numpy openpyxl
    pause
    exit
)

echo [OK] Dependencias instaladas!
echo.
echo Iniciando aplicativo...
echo.

python app.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao executar o aplicativo
    echo.
    echo Tente executar manualmente: python app.py
    pause
)
