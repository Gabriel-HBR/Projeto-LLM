"""
Prepara dados para Transfer Learning com mensagens_X_coletadas.xlsx
"""
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os

# Caminhos
LABELED_FILE = "model_training/data/processed/mensagens_rotuladas.json"
TRAIN_FILE = "model_training/data/processed/train.json"
VAL_FILE = "model_training/data/processed/val.json"
TEST_FILE = "model_training/data/processed/test.json"

print("=" * 60)
print("PREPARANDO DADOS PARA TRANSFER LEARNING")
print("=" * 60)

# Carregar labels
print("\n1. Carregando mensagens rotuladas...")
if not os.path.exists(LABELED_FILE):
    print(f"ERRO: Arquivo {LABELED_FILE} nao encontrado!")
    print("Execute primeiro: python label_data.py --auto")
    exit(1)

with open(LABELED_FILE, 'r', encoding='utf-8') as f:
    labels = json.load(f)

# Carregar mensagens originais
df = pd.read_excel("model_training/data/raw/mensagens_X_coletadas.xlsx")
messages = df['Mensagem'].tolist()

print(f"   OK! {len(labels)} mensagens rotuladas")

# Criar dataset
print("\n2. Criando dataset...")
data = []
for idx, label in labels.items():
    idx = int(idx)
    if idx < len(messages):
        data.append({
            'text': messages[idx],
            'label': label
        })

df = pd.DataFrame(data)
print(f"   OK! {len(df)} exemplos criados")

# Estatísticas
toxic = df[df['label'] == 'TOXICA'].shape[0]
non_toxic = df[df['label'] == 'NAO_TOXICA'].shape[0]
print(f"\n   TOXICAS: {toxic} ({toxic/len(df)*100:.1f}%)")
print(f"   NAO TOXICAS: {non_toxic} ({non_toxic/len(df)*100:.1f}%)")

# Balancear dataset (opcional)
print("\n3. Balanceando dataset...")
min_count = min(toxic, non_toxic)
df_toxic = df[df['label'] == 'TOXICA'].sample(n=min_count, random_state=42)
df_non_toxic = df[df['label'] == 'NAO_TOXICA'].sample(n=min_count, random_state=42)
df_balanced = pd.concat([df_toxic, df_non_toxic]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   OK! {len(df_balanced)} exemplos balanceados")

# Dividir em treino/val/teste
print("\n4. Dividindo em treino/validacao/teste (70/15/15)...")
train_df, temp_df = train_test_split(df_balanced, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"   Treino: {len(train_df)} exemplos")
print(f"   Validacao: {len(val_df)} exemplos")
print(f"   Teste: {len(test_df)} exemplos")

# Criar prompts para o modelo
def create_prompt(text, label=None):
    """Cria prompt no formato de chat"""
    system_prompt = "Voce e um classificador de toxicidade. Analise a mensagem e responda apenas TOXICA ou NAO_TOXICA."
    
    if label:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classifique: {text}"},
                {"role": "assistant", "content": label}
            ]
        }
    else:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classifique: {text}"}
            ]
        }

# Salvar dados
print("\n5. Salvando arquivos de treino...")

os.makedirs("model_training/data/processed", exist_ok=True)

# Treino
with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    for _, row in train_df.iterrows():
        json.dump(create_prompt(row['text'], row['label']), f, ensure_ascii=False)
        f.write('\n')
print(f"   OK! {TRAIN_FILE}")

# Validação
with open(VAL_FILE, 'w', encoding='utf-8') as f:
    for _, row in val_df.iterrows():
        json.dump(create_prompt(row['text'], row['label']), f, ensure_ascii=False)
        f.write('\n')
print(f"   OK! {VAL_FILE}")

# Teste
with open(TEST_FILE, 'w', encoding='utf-8') as f:
    for _, row in test_df.iterrows():
        json.dump(create_prompt(row['text'], row['label']), f, ensure_ascii=False)
        f.write('\n')
print(f"   OK! {TEST_FILE}")

print("\n" + "=" * 60)
print("DADOS PREPARADOS COM SUCESSO!")
print("=" * 60)
print("\nProximo passo: python train_transfer_learning.py")


