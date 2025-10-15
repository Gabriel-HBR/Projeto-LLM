"""
Script para rotular os dados do mensagens_X_coletadas.xlsx
"""
import pandas as pd
import json
import os

# Caminhos
XLSX_FILE = "model_training/data/raw/mensagens_X_coletadas.xlsx"
LABELED_FILE = "model_training/data/processed/mensagens_rotuladas.json"

def load_messages():
    """Carrega mensagens do Excel"""
    print("Carregando mensagens...")
    df = pd.read_excel(XLSX_FILE)
    return df['Mensagem'].tolist()

def load_existing_labels():
    """Carrega labels já existentes"""
    if os.path.exists(LABELED_FILE):
        with open(LABELED_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_labels(labels):
    """Salva labels"""
    os.makedirs(os.path.dirname(LABELED_FILE), exist_ok=True)
    with open(LABELED_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

def label_messages():
    """Interface para rotular mensagens"""
    print("=" * 60)
    print("FERRAMENTA DE ROTULACAO DE MENSAGENS")
    print("=" * 60)
    print("\nInstrucoes:")
    print("  [T] = TOXICA")
    print("  [N] = NAO TOXICA")
    print("  [P] = PULAR")
    print("  [S] = SAIR E SALVAR")
    print("=" * 60)
    
    messages = load_messages()
    labels = load_existing_labels()
    
    print(f"\nTotal de mensagens: {len(messages)}")
    print(f"Ja rotuladas: {len(labels)}")
    print(f"Faltam: {len(messages) - len(labels)}\n")
    
    for i, msg in enumerate(messages):
        # Pular se já rotulada
        if str(i) in labels:
            continue
        
        print(f"\n[{i+1}/{len(messages)}] Mensagem:")
        print(f"'{msg}'")
        print("\nClassificacao: ", end='')
        
        choice = input().strip().upper()
        
        if choice == 'T':
            labels[str(i)] = 'TOXICA'
            print("-> TOXICA")
        elif choice == 'N':
            labels[str(i)] = 'NAO_TOXICA'
            print("-> NAO TOXICA")
        elif choice == 'P':
            print("-> PULADO")
            continue
        elif choice == 'S':
            print("\nSalvando e saindo...")
            break
        else:
            print("Opcao invalida! Pulando...")
            continue
    
    save_labels(labels)
    print(f"\n{len(labels)} mensagens rotuladas salvas!")
    print(f"Arquivo: {LABELED_FILE}")

def auto_label_with_keywords():
    """Rotulação automática inicial usando palavras-chave"""
    print("=" * 60)
    print("ROTULACAO AUTOMATICA INICIAL")
    print("=" * 60)
    
    messages = load_messages()
    labels = {}
    
    # Palavras-chave tóxicas
    toxic_keywords = [
        'puto', 'puta', 'caralho', 'merda', 'idiota', 'burro', 'estupido',
        'imbecil', 'otario', 'desgraca', 'corno', 'viado', 'bicha',
        'lixo', 'vagabundo', 'fdp', 'filho da puta', 'foder', 'fuder'
    ]
    
    print("\nAnalisando mensagens...")
    for i, msg in enumerate(messages):
        msg_lower = msg.lower()
        
        # Verificar se contém palavras tóxicas
        toxic_count = sum(1 for word in toxic_keywords if word in msg_lower)
        
        if toxic_count >= 1:
            labels[str(i)] = 'TOXICA'
        else:
            labels[str(i)] = 'NAO_TOXICA'
        
        if (i + 1) % 100 == 0:
            print(f"Processadas: {i+1}/{len(messages)}")
    
    save_labels(labels)
    
    # Estatísticas
    toxic = sum(1 for v in labels.values() if v == 'TOXICA')
    non_toxic = len(labels) - toxic
    
    print(f"\n{len(labels)} mensagens rotuladas automaticamente!")
    print(f"  TOXICAS: {toxic} ({toxic/len(labels)*100:.1f}%)")
    print(f"  NAO TOXICAS: {non_toxic} ({non_toxic/len(labels)*100:.1f}%)")
    print(f"\nArquivo salvo: {LABELED_FILE}")
    print("\nDica: Revise as rotulacoes com 'python label_data.py --review'")

def show_stats():
    """Mostra estatísticas das rotulações"""
    labels = load_existing_labels()
    
    if not labels:
        print("Nenhuma mensagem rotulada ainda!")
        return
    
    toxic = sum(1 for v in labels.values() if v == 'TOXICA')
    non_toxic = len(labels) - toxic
    
    print("=" * 60)
    print("ESTATISTICAS DE ROTULACAO")
    print("=" * 60)
    print(f"\nTotal rotulado: {len(labels)}")
    print(f"  TOXICAS: {toxic} ({toxic/len(labels)*100:.1f}%)")
    print(f"  NAO TOXICAS: {non_toxic} ({non_toxic/len(labels)*100:.1f}%)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            auto_label_with_keywords()
        elif sys.argv[1] == '--stats':
            show_stats()
        elif sys.argv[1] == '--review':
            label_messages()
        else:
            print("Uso:")
            print("  python label_data.py --auto    # Rotulacao automatica")
            print("  python label_data.py --review  # Revisar/rotular manualmente")
            print("  python label_data.py --stats   # Ver estatisticas")
    else:
        print("=" * 60)
        print("ROTULADOR DE MENSAGENS")
        print("=" * 60)
        print("\nOpcoes:")
        print("  1. Rotulacao automatica (usando palavras-chave)")
        print("  2. Rotulacao manual")
        print("  3. Ver estatisticas")
        print("\nEscolha: ", end='')
        
        choice = input().strip()
        
        if choice == '1':
            auto_label_with_keywords()
        elif choice == '2':
            label_messages()
        elif choice == '3':
            show_stats()
        else:
            print("Opcao invalida!")


