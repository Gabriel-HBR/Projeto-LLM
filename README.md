# Detector de Toxicidade em Mensagens com Transfer Learning

Sistema inteligente para identificar conteúdo tóxico em mensagens de texto usando **Transfer Learning** com TinyLlama

## Descrição

Este projeto oferece duas abordagens para classificação de toxicidade:

1. **Modo Rápido**: Classificador baseado em palavras-chave 
2. **Transfer Learning**: Modelo LLM treinado com seus dados 

## Início Rápido

### **Windows - Modo Automático (Mais Fácil!)**

Clique em um dos arquivos `.bat`:

- **`ESCOLHA_O_MODO.bat`** → Menu interativo para escolher o modo
- **`EXECUTAR_RAPIDO.bat`** → Modo Rápido (5 min)
- **`EXECUTAR_TRANSFER_LEARNING.bat`** → Modo Completo (varia conforme hardware)

### **Modo 1: Uso Imediato (Recomendado para começar)**

```bash
cd Projeto-LLM
pip install pandas numpy openpyxl
python app.py
```

### **Modo 2: Transfer Learning Completo (Um único comando no Windows)**

```bash
# Windows: Clique em EXECUTAR_TRANSFER_LEARNING.bat
# Ou execute manualmente:

# 1. Instalar dependências
pip install pandas numpy openpyxl torch transformers datasets scikit-learn

# 2. Rotular dados automaticamente
python label_data.py --auto

# 3. Preparar dados
python prepare_data_transfer_learning.py

# 4. Treinar modelo
python train_transfer_learning.py

# 5. Usar o app
python app.py
```

**Parâmetros de Treinamento:**
- **Modelo**: TinyLlama-1.1B-Chat-v1.0
- **Épocas**: 3
- **Batch Size**: 8
- **Learning Rate**: 2e-3
- **Max Length**: 256 tokens

## Como Funciona o Transfer Learning

### Etapa 1: Rotulação dos Dados

```
mensagens_X_coletadas.xlsx (1000 mensagens)
              ↓
    label_data.py --auto
              ↓
  mensagens_rotuladas.json
```

O sistema usa palavras-chave para rotular automaticamente:
- **Tóxicas**: Mensagens com insultos, palavrões, ofensas
- **Não-tóxicas**: Mensagens normais

### Etapa 2: Preparação

```
mensagens_rotuladas.json
              ↓
prepare_data_transfer_learning.py
              ↓
Treino (275) + Validação (59) + Teste (60)
(Dataset balanceado formatado em chat template)
```

**Formato Chat Template:**
O modelo usa um formato especial de conversa:
```
<|system|>
Você é um classificador de toxicidade. Responda apenas com TÓXICA ou NAO_TOXICA
</s>
<|user|>
Você é um idiota
</s>
<|assistant|>
TÓXICA
</s>
```

### Etapa 3: Transfer Learning

```
TinyLlama-1.1B-Chat-v1.0 (modelo base pré-treinado)
              ↓
    Fine-tuning (3 épocas, batch=8, LR=2e-3)
              ↓
  Modelo personalizado treinado
              ↓
  Salvo em: models/toxicity_transfer_learning/
```

**Configurações de Treinamento:**
- Max Length: 256 tokens
- Gradient Accumulation: 2 steps (efetivamente batch=16)
- Warmup Steps: 50
- Evaluation Strategy: Every 100 steps
- Precision: FP16 (GPU) / FP32 (CPU)
- Data Collator: LanguageModeling (MLM=False para causal LM)
- Chat Template: Formata automaticamente conversas
- Device: Auto (detecta GPU/CPU automaticamente)

### Etapa 4: Uso

```
Nova mensagem → Modelo → TÓXICA ou NÃO TÓXICA
```

## Funcionalidades

- **Interface gráfica bonita** (Tkinter)
- **Rotulação automática** de dados
- **Transfer Learning** com TinyLlama
- **Classificação em tempo real**
- **Confiança da predição**
- **Dataset brasileiro** (mensagens_X_coletadas.xlsx)
- **Funciona offline** (após treinamento)

## Interface do Aplicativo

A interface possui:
- **Indicador do modelo** (topo): Mostra qual modelo está sendo usado
- **Seletor de modelo** (se TL disponível): Troca entre modelos em tempo real
- Campo de texto para mensagem
- Botão "Analisar Mensagem"
- Resultado visual: SEGURA ou TÓXICA
- Percentual de confiança
- Cores intuitivas (verde/vermelho)
- **Dica contextual** (rodapé): Instruções contextuais

**Seletor de Modelo:**
```
Selecionar Modelo:
   (•) Modo Rápido    ( ) Transfer Learning
```
- Troque entre modelos **sem fechar o app**!
- Aparece automaticamente quando Transfer Learning está disponível
- **Modo Híbrido**: TL usa classificador simples como fallback para máxima robustez

**Indicador de Status:**
```
Modelo Ativo: Classificador Rápido (Regex + 150+ padrões)
```

## Arquitetura da Rede Neural

### O que é uma Rede Neural?

Imagine que você quer ensinar um computador a entender textos. Uma **rede neural** é como o cérebro artificial que faz isso - ela é composta de "neurônios" conectados que processam informação.

### Como Funciona Nossa Rede

A nossa rede usa a arquitetura **Transformer** (como o ChatGPT). Vamos entender cada parte:

#### 1. Embedding (Conversão de Palavras em Números)

**O que é?**
- Palavras em português precisam virar números para o computador entender
- Cada palavra vira um "vetor" (lista de números) que representa seu significado

**Exemplo simples:**
```
"você" → [0.13, 0.92, -0.44, 0.15, ...]
"idiota" → [0.90, -0.74, 1.12, 0.03, ...]
```

**Por que 2048 dimensões?**
- Quanto mais números, mais detalhada é a representação da palavra
- 2048 é o tamanho escolhido para capturar nuances e significados

#### 2. Atenção (Multi-Head Self-Attention)

**O que é?**
- É como o modelo "presta atenção" em palavras importantes
- Quando lê "Você é um idiota", a atenção foca em "você" e "idiota" ao mesmo tempo

**Analogia:**
Imagine lendo um texto e grifando as palavras importantes. A rede faz isso automaticamente, focando onde está a toxicidade.

**16 cabeças de atenção:**
- O modelo olha o texto de 16 jeitos diferentes ao mesmo tempo
- Cada "olho" captura um tipo diferente de relação entre palavras
- É como ter 16 pessoas analisando a mensagem simultaneamente

#### 3. Feed-Forward (Rede Neural Simples)

**O que é?**
- Redes que processam as informações recebidas
- Após a atenção, essas redes refinam o entendimento

**Analogia:**
Imagine que após identificar palavras importantes, você precisa "pensar" sobre elas. Feed-Forward faz esse "pensamento".

**5632 neurônios:**
- É a capacidade de processamento
- Mais neurônios = mais capacidade de entender nuances

#### 4. Normalização e Conexões Residuais

**Normalização:**
- Estabiliza os números para evitar desequilíbrios
- É como balancear os ingredientes de uma receita

**Skip Connections:**
- Mantém conexões diretas entre camadas distantes
- Permite que informações importantes não se percam no meio do caminho
- É como ter atalhos em um edifício com muitos andares

### Estrutura Completa do Modelo

```
Mensagem: "Você é um idiota"
    ↓
PASSO 1: Vira números
    ↓
[você: 0.13, -0.44, ...]
[é: 0.02, 0.71, ...]
[um: -0.22, 0.88, ...]
[idiota: 0.90, -0.74, ...]
    ↓
PASSO 2: 22 camadas de processamento
(Cada camada: Atenção → Feed-Forward)
    ↓
    Camada 1: Analisa relação das palavras
    Camada 2: Refina o entendimento
    ...
    Camada 22: Compreensão profunda do texto
    ↓
PASSO 3: Classificação final
    ↓
Resultado: TÓXICA (94.3% de certeza)
```

### Números da Nossa Rede

- **22 camadas**: O texto passa por 22 "processamentos" diferentes
- **1.1 bilhão de parâmetros**: "Memória" da rede - tudo que ela aprendeu
- **2048 dimensões**: Tamanho dos vetores que representam palavras
- **16 cabeças de atenção**: O modelo olha o texto de 16 ângulos

### Por que Usar Essa Arquitetura?

1. **Transformer é o estado da arte**: Usado nos melhores modelos de IA
2. **Atenção captura contexto**: Entende relação entre palavras distantes
3. **Transfer Learning**: Reaproveita conhecimento já aprendido
4. **Eficiente**: Funciona até em computadores sem GPU

## Como o Modelo Calcula as Probabilidades

### 1. O que o modelo realmente faz

O modelo (como o GPT) é essencialmente uma rede neural com camadas Transformer. Cada camada analisa o texto de entrada e transforma as palavras em vetores numéricos (chamados **embeddings**), que carregam o significado e o contexto.

**Exemplo:**

| Palavra | Vetor interno (simplificado) |
|---------|-------------------------------|
| você | [0.13, 0.92, -0.44, ...] |
| é | [0.02, 0.71, -0.32, ...] |
| um | [-0.22, 0.88, -0.11, ...] |
| idiota | [0.90, -0.74, 1.12, ...] |

Esses vetores passam por várias camadas que capturam:
- **Relações entre palavras** (ex: "você" + "idiota" → ofensa dirigida);
- **Tom emocional e semântico**;
- **Contexto da frase completa**.

### 2. Logits — a pontuação antes das probabilidades

Após o texto ser processado, a última camada gera valores brutos chamados **logits** — um número para cada possível rótulo (ou token de saída).

**Exemplo:**

| Classe (rótulo) | Logit |
|----------------|-------|
| TOXICA | 5.2 |
| NAO_TOXICA | -1.8 |
| MUITO | -3.1 |
| TALVEZ | -3.5 |

Esses valores não são probabilidades ainda — são apenas pontuações relativas: **quanto maior o logit, mais confiante o modelo está naquela opção**.

### 3. Aplicando o Softmax

O **softmax** transforma esses logits em probabilidades que somam 1. A fórmula é:

```
P_i = e^z_i / Σ(e^z_j)
```

Onde:
- `z_i` = logit de cada classe
- `e^z_i` = exponencial do logit (para amplificar diferenças)

**Calculando o exemplo:**

```
e^5.2 = 181.27
e^-1.8 = 0.165
e^-3.1 = 0.045
e^-3.5 = 0.030

Denominador = 181.27 + 0.165 + 0.045 + 0.030 = 181.51

P(TOXICA) = 181.27 / 181.51 = 0.9987 ≈ 99.87%
```

No seu exemplo, arredondou para 94.3%, o que seria o caso se os logits tivessem diferenças um pouco menores.

### 4. Interpretação semântica

Esses números vêm do aprendizado do modelo, que foi treinado com milhões de exemplos rotulados (textos ofensivos e não ofensivos, neste caso). Assim, os pesos da rede foram ajustados para que padrões como *"você é um idiota"* resultem em logits altos para "TOXICA".

O modelo não "entende" moralmente que é uma ofensa — ele apenas reconhece um padrão estatístico fortemente correlacionado com textos ofensivos.

### 5. Escolha final e confiança

O modelo escolhe o rótulo com a maior probabilidade. Essa probabilidade é interpretada como confiança (embora tecnicamente não seja uma confiança humana, e sim uma medida relativa de plausibilidade entre opções).

### Resumo visual

```
Texto: "Você é um idiota"

    ↓ Tokenização
[você, é, um, idiota]

    ↓ Embeddings + Attention Layers
[você] contextualizado por [idiota]

    ↓ Saída (logits)
[TOXICA=5.2, NAO_TOXICA=-1.8, MUITO=-3.1, TALVEZ=-3.5]

    ↓ Softmax
[94.3%, 4.1%, 0.8%, 0.5%]

    ↓ Resultado final
"TOXICA" (Confiança: 94.3%)
```

## Exemplos

### Mensagens Seguras:
```
"Bom dia! Como você está?"          → NÃO TÓXICA
"Adorei esse filme, muito bom!"     → NÃO TÓXICA
"Vamos marcar um café amanhã?"      → NÃO TÓXICA
```

### Mensagens Tóxicas:
```
"Você é um idiota"                  → TÓXICA
"Vai se f*der seu lixo"             → TÓXICA
"Você é um verme babaca"            → TÓXICA
"Seu incompetente nojento"          → TÓXICA
```

## Comandos Úteis

```bash
# Rotulação
python label_data.py --auto      # Rotular automaticamente
python label_data.py --review    # Revisar manualmente
python label_data.py --stats     # Ver estatísticas

# Preparação e treinamento
python prepare_data_transfer_learning.py
python train_transfer_learning.py

# Testes
python test_classifier.py        # Testar classificador

# Executar app
python app.py
```

## O que é Transfer Learning?

**Transfer Learning** é uma técnica onde pegamos um modelo já treinado (TinyLlama) e o ajustamos para nossa tarefa específica (detectar toxicidade).

### Vantagens:
- Não precisa treinar do zero
- Aprende com poucos dados
- Alta precisão
- Entende contexto e nuances

### Por que TinyLlama?
- **Modelo**: TinyLlama-1.1B-Chat-v1.0
- Pequeno (~1.1B parâmetros)
- Rápido para treinar (otimizado para GPU e CPU)
- Boa precisão em português
- Funciona em CPU (mais lento) ou GPU (recomendado)
- Já vem com chat template otimizado

## Requisitos

### Modo Rápido:
- Python 3.8+
- pandas, numpy, openpyxl
- ~10MB espaço

### Transfer Learning:
- Python 3.8+
- torch, transformers, datasets
- ~2GB espaço (modelo + dependências)
- GPU recomendado (mas funciona em CPU)
- **Modelo Base**: TinyLlama-1.1B-Chat-v1.0
- **Tensorização**: FP16 (GPU) ou FP32 (CPU)

## Solução de Problemas

### "No module named 'torch'"
```bash
pip install torch transformers
```

### "No module named 'tkinter'"
- **Windows/Mac**: Já vem com Python
- **Linux**: `sudo apt-get install python3-tk`

### Treinamento lento (CPU)
1. Use apenas o modo rápido: `python app.py`
2. Ou reduza epochs: edite `train_transfer_learning.py` → `EPOCHS = 1`
3. Para treinar mais rápido em CPU, altere também:
   ```python
   BATCH_SIZE = 4  
   ```

### Erro de memória (GPU)
Reduza batch size em `train_transfer_learning.py`:
```python
BATCH_SIZE = 2  # ou 1
```

### Dataset não encontrado
Certifique-se de que existe:
```
model_training/data/raw/mensagens_X_coletadas.xlsx
```

## Dataset

Utiliza `mensagens_X_coletadas.xlsx`:
- 1000 mensagens em português brasileiro
- Coletadas do Twitter/X
- Rotuladas automaticamente 

## Integrantes

- **ERICK EIJI NAGAO** - RA: 21.00690-3
- **IGOR IMPROTA MARTINEZ DA SILVA** - RA: 21.00834-5
- **GABRIEL HENRIQUE BACA RADO** - RA: 21.01286-5
- **RYUSKE HIDEAKI SATO** - RA: 21.00745-4
- **VINICIUS DE OLIVEIRA BERTI** - RA: 21.01219-9

**Desenvolvido usando Python, TinyLlama, Transformers e Tkinter**

*Projeto educacional de código aberto para detecção de toxicidade em mensagens*

