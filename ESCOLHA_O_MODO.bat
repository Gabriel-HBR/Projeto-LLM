@echo off
chcp 65001 > nul
cls

:menu
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║     🛡️  DETECTOR DE TOXICIDADE EM MENSAGENS                 ║
echo ║     Escolha o Modo de Operacao                              ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │  ESCOLHA UMA OPCAO:                                          │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] MODO RAPIDO
echo       └─ Classificacao por palavras-chave
echo       └─ Instalacao: ~2 minutos
echo       └─ Recomendado para: Testes e demonstracoes
echo.
echo   [2] MODO TRANSFER LEARNING
echo       └─ Treina modelo LLM personalizado
echo       └─ Instalacao + Treino: ~1-3 horas
echo       └─ Recomendado para: Uso profissional
echo.
echo   [3] APENAS ABRIR O APP
echo       └─ Abre o aplicativo sem instalar nada
echo       └─ Use se ja instalou anteriormente
echo.
echo   [0] SAIR
echo.
echo ══════════════════════════════════════════════════════════════
echo.
choice /C 1230 /N /M "Digite sua escolha [1, 2, 3 ou 0]: "

if errorlevel 4 goto exit
if errorlevel 3 goto app_only
if errorlevel 2 goto transfer_learning
if errorlevel 1 goto modo_rapido

:modo_rapido
cls
echo.
echo [MODO RAPIDO SELECIONADO]
echo.
call EXECUTAR_RAPIDO.bat
goto menu

:transfer_learning
cls
echo.
echo [TRANSFER LEARNING SELECIONADO]
echo.
call EXECUTAR_TRANSFER_LEARNING.bat
goto menu

:app_only
cls
echo.
echo [ABRINDO APLICATIVO]
echo.
python app.py
if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao executar. Instale as dependencias primeiro!
    pause
)
goto menu

:exit
cls
echo.
echo Obrigado por usar o Detector de Toxicidade!
echo.
timeout /t 2 >nul
exit



