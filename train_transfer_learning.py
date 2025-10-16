"""
Transfer Learning com TinyLlama usando mensagens_X_coletadas.xlsx
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilitar warnings

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

# Configurações
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "models/toxicity_transfer_learning"
TRAIN_FILE = "model_training/data/processed/train.json"
VAL_FILE = "model_training/data/processed/val.json"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-4

print("=" * 60)
print("TRANSFER LEARNING - CLASSIFICADOR DE TOXICIDADE")
print("=" * 60)

# Verificar arquivos
if not os.path.exists(TRAIN_FILE):
    print(f"\nERRO: {TRAIN_FILE} nao encontrado!")
    print("Execute primeiro:")
    print("  1. python label_data.py --auto")
    print("  2. python prepare_data_transfer_learning.py")
    exit(1)

# Verificar GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n1. Dispositivo: {device}")
if device == "cpu":
    print("   AVISO: Treinamento em CPU sera LENTO!")
    print("   Recomenda-se usar GPU ou reduzir epochs")

# Carregar dados
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
print("   OK! Modelo carregado")

# Preparar dados
print("\n4. Preparando dados para fine-tuning...")

def format_chat(messages):
    """Formata mensagens em texto"""
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
    texts = [format_chat(ex["messages"]) for ex in examples["data"]]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
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

# Configurar treinamento
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
    fp16=device == "cuda",
    load_best_model_at_end=True,
    report_to="none",
    gradient_accumulation_steps=2,
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

# Treinar
print("\n6. Iniciando Fine-Tuning...")
print("   (Isso vai demorar alguns minutos...)")
print("\n" + "-" * 60)

try:
    trainer.train()
    print("-" * 60)
    print("\n   OK! Treinamento concluido")
except KeyboardInterrupt:
    print("\n\n   AVISO: Treinamento interrompido pelo usuario")
except Exception as e:
    print(f"\n\n   ERRO: {e}")
    exit(1)

# Salvar modelo
print("\n7. Salvando modelo treinado...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   OK! Modelo salvo em: {OUTPUT_DIR}")

# Avaliar
print("\n8. Avaliando modelo...")
eval_results = trainer.evaluate()
print(f"   Loss de validacao: {eval_results['eval_loss']:.4f}")

print("\n" + "=" * 60)
print("TRANSFER LEARNING CONCLUIDO!")
print("=" * 60)
print(f"\nModelo salvo em: {OUTPUT_DIR}")
print("\nProximo passo: Atualizar app.py para usar o modelo treinado")


