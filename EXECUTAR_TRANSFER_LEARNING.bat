@echo off
chcp 65001 > nul
cls

echo.
echo -----------------------------------------------------------------
echo  TRANSFER LEARNING - MODO COMPLETO                       
echo  Detector de Toxicidade com TinyLlama                    
echo -----------------------------------------------------------------
echo.
echo AVISO: Este processo pode demorar de 5 a 7 horas!
echo        
echo.
echo Pressione Ctrl+C para cancelar ou
pause

cls
echo.
echo -----------------------------------------------------------------
echo   INICIANDO PROCESSO DE TRANSFER LEARNING
echo -----------------------------------------------------------------
echo.

REM -----------------------------------------------------------------   
REM ETAPA 1: Verificar e instalar dependências
REM -----------------------------------------------------------------

echo [ETAPA 1/5] Instalando dependencias...
echo.
echo Isso pode demorar alguns minutos na primeira vez...
echo.

pip install pandas numpy openpyxl --quiet 2>nul
pip install torch transformers datasets scikit-learn accelerate --quiet 2>nul

if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao instalar dependencias!
    echo.
    echo Tente instalar manualmente:
    echo   pip install pandas numpy openpyxl torch transformers datasets scikit-learn
    echo.
    pause
    exit /b 1
)

echo [OK] Dependencias instaladas!
echo.

REM -----------------------------------------------------------------   
REM ETAPA 2: Rotular dados automaticamente
REM -----------------------------------------------------------------

echo -----------------------------------------------------------------
echo [ETAPA 2/5] Rotulando mensagens automaticamente...
echo -----------------------------------------------------------------
echo.

if exist "model_training\data\processed\mensagens_rotuladas.json" (
    echo [INFO] Arquivo de labels ja existe!
    echo.
    choice /C SN /M "Deseja re-rotular os dados? (S=Sim, N=Nao)"
    if errorlevel 2 goto skip_label
)

python label_data.py --auto

if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao rotular dados!
    pause
    exit /b 1
)

echo.
echo [OK] Dados rotulados com sucesso!
echo.
goto continue_label

:skip_label
echo [INFO] Pulando rotulacao...
echo.

:continue_label

REM -----------------------------------------------------------------
REM ETAPA 3: Preparar dados para treinamento
REM -----------------------------------------------------------------

echo -----------------------------------------------------------------
echo [ETAPA 3/5] Preparando dados para treinamento...
echo -----------------------------------------------------------------
echo.

python prepare_data_transfer_learning.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha ao preparar dados!
    pause
    exit /b 1
)

echo.
echo [OK] Dados preparados com sucesso!
echo.

REM -----------------------------------------------------------------
REM ETAPA 4: Treinar modelo (Transfer Learning)
REM -----------------------------------------------------------------

echo -----------------------------------------------------------------
echo [ETAPA 4/5] TREINANDO MODELO COM TRANSFER LEARNING
echo -----------------------------------------------------------------
echo.
echo ATENCAO: Esta etapa pode demorar bastante!
echo.
echo Voce pode parar o treinamento a qualquer momento com Ctrl+C
echo (o progresso sera salvo).
echo.
pause

echo.
echo Iniciando treinamento...
echo.

python train_transfer_learning.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha durante o treinamento!
    echo.
    echo Possiveis causas:
    echo   - Falta de memoria (reduza BATCH_SIZE)
    echo   - GPU sem espaco (use CPU)
    echo   - Interrupcao manual (Ctrl+C)
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Modelo treinado com sucesso!
echo.

REM -----------------------------------------------------------------
REM ETAPA 5: Executar aplicativo
REM -----------------------------------------------------------------

echo -----------------------------------------------------------------
echo [ETAPA 5/5] Iniciando aplicativo...
echo -----------------------------------------------------------------
echo.

choice /C SN /M "Deseja abrir o aplicativo agora? (S=Sim, N=Nao)"
if errorlevel 2 goto skip_app

python app.py

:skip_app

REM -----------------------------------------------------------------
REM CONCLUSÃO
REM -----------------------------------------------------------------

cls
echo.
echo -----------------------------------------------------------------
echo                                                          
echo  ✓ TRANSFER LEARNING CONCLUIDO COM SUCESSO!             
echo                                                          
echo -----------------------------------------------------------------
echo.
echo Modelo treinado salvo em:
echo   models/toxicity_transfer_learning/
echo.
echo Para usar o aplicativo:
echo   - Execute: EXECUTAR_AQUI.bat
echo   - Ou: python app.py
echo.
echo Para testar o modelo treinado:
echo   python test_transfer_learning.py
echo.
echo -----------------------------------------------------------------
echo.
pause





