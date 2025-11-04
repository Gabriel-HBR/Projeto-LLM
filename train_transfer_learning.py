"""
Transfer Learning com TinyLlama usando mensagens_X_coletadas.xlsx

- Objetivo: ajustar (fine-tuning) o modelo base TinyLlama-1.1B-Chat-v1.0 para
  classificar toxicidade em mensagens de texto em pt-BR, retornando rótulos
  como "TÓXICA" ou "NÃO TÓXICA".
- Pipeline completo:
  1) Rotulação automática de dados: `python label_data.py --auto`
     - Gera `mensagens_rotuladas.json` a partir de planilha coletada
  2) Preparação dos dados para TL: `python prepare_data_transfer_learning.py`
     - Cria `model_training/data/processed/train.json` e `val.json`
     - Formata conversas no template de chat do modelo
     - Balanceia e separa treino/validação/teste
  3) Treinamento (este script): `python train_transfer_learning.py`
     - Ajusta o TinyLlama com base nos dados em formato de chat
  4) Uso do app: `python app.py`

Detalhes importantes usados aqui:
- Modelo base: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (transformer causal LM)
- Comprimento máximo (MAX_LENGTH): 256 tokens (equilíbrio custo/contexto)
- Hiperparâmetros típicos: épocas=3, batch=4 (acumulação de gradiente para
  efetivo maior), learning rate=2e-3, warmup_steps=50
- Precisão: FP16 em GPU (quando suportado), FP32 em CPU (mais estável)
- Data collator: LanguageModeling (mlm=False) por ser causal LM
- Dispositivo: detecção automática (GPU/CPU) e mapeamento de camadas
- Saída: modelo salvo em `models/toxicity_transfer_learning/`

Sobre o template de chat (seguindo README):
<|system|> Você é um classificador de toxicidade. Responda apenas com TÓXICA ou NAO_TOXICA </s>
<|user|>   [mensagem do usuário]                                                                </s>
<|assistant|> [resposta/predição esperada]                                                      </s>

Observações práticas (troubleshooting):
- Treinamento em CPU é lento: reduza `EPOCHS` para 1 e/ou `BATCH_SIZE` para 2/1
- OOM em GPU: reduza `BATCH_SIZE` e/ou `MAX_LENGTH`
- Certifique-se que os arquivos `train.json` e `val.json` existem (gerados na
  etapa de preparação) antes de treinar
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilitar warnings do OneDNN em algumas instalações

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Configurações principais do treinamento
# - MODEL_NAME: checkpoint base pré-treinado (TinyLlama chat) adequado para
#   conversas em português com template de chat
# - OUTPUT_DIR: onde o modelo ajustado será salvo
# - TRAIN/VAL_FILE: gerados por `prepare_data_transfer_learning.py` em formato
#   JSONL (uma linha por exemplo), cada exemplo contendo `messages`
# - MAX_LENGTH: limita tokens por exemplo (memória/tempo vs contexto)
# - BATCH_SIZE/EPOCHS/LEARNING_RATE: hiperparâmetros alinhados ao README
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "models/toxicity_transfer_learning"
TRAIN_FILE = "model_training/data/processed/train.json"
VAL_FILE = "model_training/data/processed/val.json"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-3

print("=" * 60)
print("TRANSFER LEARNING - CLASSIFICADOR DE TOXICIDADE")
print("=" * 60)

# Verificar arquivos de treino criados na etapa de preparação
# Dica (README): executar antes
#   1) python label_data.py --auto
#   2) python prepare_data_transfer_learning.py
if not os.path.exists(TRAIN_FILE):
    print(f"\nERRO: {TRAIN_FILE} nao encontrado!")
    print("Execute primeiro:")
    print("  1. python label_data.py --auto")
    print("  2. python prepare_data_transfer_learning.py")
    exit(1)

# Verificar hardware disponível
# - Em GPU: treinos significativamente mais rápidos; dtype preferível fp16
# - Em CPU: funciona, porém lento; README sugere reduzir épocas/batch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n1. Dispositivo: {device}")
if device == "cpu":
    print("   AVISO: Treinamento em CPU sera LENTO!")
    print("   Recomenda-se usar GPU ou reduzir epochs")

# Carregar dados de treino/validação (formato JSONL gerado na preparação)
print("\n2. Carregando dados de treino...")
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl(TRAIN_FILE)
val_data = load_jsonl(VAL_FILE)
print(f"   Treino: {len(train_data)} exemplos")
print(f"   Validacao: {len(val_data)} exemplos")

# Carregar modelo e tokenizer
# - trust_remote_code=True: necessário para alguns modelos chat que definem
#   templates/comportamentos customizados
# - pad_token = eos_token: para permitir padding consistente em causal LM
print(f"\n3. Carregando modelo base: {MODEL_NAME}")
print("   (Isso pode demorar na primeira vez...)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True
)
# IMPORTANTE (Arquitetura interna do Transformer):
# - O TinyLlama implementa internamente, em cada bloco Transformer:
#   1) Normalização (LayerNorm)
#   2) Self-Attention com múltiplas cabeças (Multi-Head Self-Attention)
#      → calcula atenção em paralelo em várias "cabeças" e concatena
#   3) Conexão residual (skip connection) somando entrada + saída da atenção
#   4) Feed-Forward (MLP) aplicado posição-a-posicão
#   5) Nova normalização e conexão residual após o MLP
#
# - Acontece durante as chamadas ao "forward" do modelo, invocadas pelo Trainer
#   no `trainer.train()` e `trainer.evaluate()`. Não implementamos manualmente
#   as camadas; elas são executadas dentro do TinyLlama carregado acima.
#
try:
    print("   Config do modelo:")
    print(f"   - num_hidden_layers: {getattr(model.config, 'num_hidden_layers', 'desconhecido')}")
    print(f"   - num_attention_heads: {getattr(model.config, 'num_attention_heads', 'desconhecido')}")
    print(f"   - hidden_size: {getattr(model.config, 'hidden_size', 'desconhecido')}")
except Exception:
    pass
print("   OK! Modelo carregado")

# Preparar dados no template de chat esperado pelo modelo
# - O README define o template com blocos <|system|>, <|user|>, <|assistant|>
# - Cada mensagem termina com </s> para delimitar segmentos
print("\n4. Preparando dados para fine-tuning...")

def format_chat(messages):
    """Formata mensagens (lista de dicts com role/content) no chat template.

    Exemplo esperado (README):
    <|system|>   Você é um classificador de toxicidade. ... </s>
    <|user|>     Você é um idiota                             </s>
    <|assistant|>TÓXICA                                        </s>
    """
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|system|>\n{content}</s>\n"
        elif role == "user":
            text += f"<|user|>\n{content}</s>\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}</s>\n"
    return text

def tokenize_function(examples):
    # Concatena cada conversa como um único texto (causal LM aprende próximo token)
    texts = [format_chat(ex["messages"]) for ex in examples["data"]]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    # Para causal LM, rótulos são os próprios input_ids deslocados internamente
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

train_dataset = Dataset.from_dict({"data": train_data})
val_dataset = Dataset.from_dict({"data": val_data})

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"]
)
val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"]
)
print("   OK! Dados tokenizados")

# Configurar treinamento (TrainingArguments)
# - gradient_accumulation_steps=4: aumenta batch efetivo sem estourar memória
# - warmup_steps=50: esquenta o otimizador para estabilizar o início do treino
# - eval/log/save por passos: monitoramento periódico e checkpoints limitados
# - fp16/bf16: desabilitado aqui por segurança; em GPUs compatíveis pode-se
#   habilitar fp16 para maior velocidade
print("\n5. Configurando Transfer Learning...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    # fp16=device == "cuda",
    fp16=False,
    bf16=False,
    fp16_full_eval=False,
    load_best_model_at_end=True,
    report_to="none",
    gradient_accumulation_steps=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("   OK! Configuracao completa")
print(f"\n   Parametros:")
print(f"   - Epocas: {EPOCHS}")
print(f"   - Batch Size: {BATCH_SIZE}")
print(f"   - Learning Rate: {LEARNING_RATE}")
print(f"   - Dispositivo: {device}")

# Treinamento
# - Pode levar minutos/horas dependendo de CPU/GPU e tamanho dos dados
print("\n6. Iniciando Fine-Tuning...")
print("   (Isso vai demorar alguns minutos...)")
print("\n" + "-" * 60)

try:
    trainer.train()
    # NOTA: É aqui que, a cada forward/backward, o modelo executa:
    # - LayerNorm → Multi-Head Self-Attention → Residual
    # - LayerNorm → Feed-Forward (MLP)       → Residual
    # em todas as camadas (blocos Transformer) do TinyLlama.
    print("-" * 60)
    print("\n   OK! Treinamento concluido")
except KeyboardInterrupt:
    print("\n\n   AVISO: Treinamento interrompido pelo usuario")
except Exception as e:
    print(f"\n\n   ERRO: {e}")
    exit(1)

# Salvar modelo
# - Saída padrão (README): models/toxicity_transfer_learning/
print("\n7. Salvando modelo treinado...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   OK! Modelo salvo em: {OUTPUT_DIR}")

# Avaliação simples (loss de validação)
# - Métrica primária aqui é a loss; para medir acurácia de classificação,
#   pode-se construir um avaliador que gere rótulos e compare com ground truth
print("\n8. Avaliando modelo...")
eval_results = trainer.evaluate()
print(f"   Loss de validacao: {eval_results['eval_loss']:.4f}")

print("\n" + "=" * 60)
print("TRANSFER LEARNING CONCLUIDO!")
print("=" * 60)
print(f"\nModelo salvo em: {OUTPUT_DIR}")
print("\nProximo passo: Atualizar app.py para usar o modelo treinado")


