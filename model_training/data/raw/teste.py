
# -*- coding: utf-8 -*-
"""
Script para corrigir problemas de codificação no arquivo ToLD-BR.csv
Corrige caracteres que foram mal codificados (UTF-8 lido como Latin-1/Windows-1252)
"""

import pandas as pd
import re
from pathlib import Path

def detect_encoding(file_path):
    """Tenta detectar a codificação do arquivo testando diferentes encodings"""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1000)  # Lê apenas os primeiros 1000 caracteres para testar
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    return 'utf-8'  # Fallback para UTF-8

def fix_encoding_issues(text):
    """
    Corrige problemas de codificação comuns em texto português
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Dicionário de correções para caracteres mal codificados
    corrections = {
        # Acentos e caracteres especiais básicos
        '锚': 'ê',  # ê mal codificado
        '茫': 'ã',  # ã mal codificado
        '铆': 'í',  # í mal codificado
        '谩': 'á',  # á mal codificado
        '贸': 'ó',  # ó mal codificado
        '茅': 'é',  # é mal codificado
        '莽': 'ç',  # ç mal codificado
        '脿': 'à',  # à mal codificado
        '么': 'ã',  # ã mal codificado
        '煤': 'ú',  # ú mal codificado
        '玫': 'é',  # é mal codificado
        '鈥': '…',  # reticências mal codificadas
        '馃憡': '😄',  # emoji sorriso
        '馃嚙': '😍',  # emoji apaixonado
        '馃嚪': '😊',  # emoji sorriso tímido
        '馃棏': '😭',  # emoji chorando
        '馃叞': '😅',  # emoji suando
        '馃槫': '😭',  # emoji chorando (outra variação)
        '馃槀': '😂',  # emoji rindo (outra variação)
        '锔忦焻帮笍': '😂',  # emoji rindo (sequência mal codificada)
        
        # Correções adicionais do notebook
        '芒': 'â',  # â mal codificado
        '么': 'ô',  # ô mal codificado
        '玫': 'õ',  # õ mal codificado
        '脳': 'x',  # x mal codificado
        '脡': 'É',  # É mal codificado
        '脥': 'Í',  # Í mal codificado
        '脙': 'Ã',  # Ã mal codificado
        '聽': ' ',  # espaço mal codificado
        '淥': 'O',  # O mal codificado
        '渉': 'H',  # H mal codificado
        '渃': 'C',  # C mal codificado
        '減': 'C',  # C mal codificado
        '楤': 'B',  # B mal codificado
        '脟': 'Ç',  # Ç mal codificado
        '脭': 'Ô',  # Ô mal codificado
        '脕': 'Á',  # Á mal codificado
        '脗': 'Â',  # Â mal codificado
        '陋': 'ª',  # ª mal codificado
        
        # Caracteres adicionais detectados automaticamente
        '馃': '',  # emoji mal codificado
        '"': '"',  # aspas inteligentes
        '"': '"',  # aspas inteligentes
        '‍': '',   # zero width joiner
        '': '',   # replacement character
        '°': '°',  # grau
        ' ': ' ',  # non-breaking space
        'º': 'º',  # ordinal masculino
        '锔': '',  # caractere mal codificado
        '嶁': '',  # caractere mal codificado
        '槏': '',  # caractere mal codificado
        '\u2019': "'",  # apostrofe inteligente
        '笍': '',  # caractere mal codificado
        '鉂': '',  # caractere mal codificado
        '従': '',  # caractere mal codificado
        '檧': '',  # caractere mal codificado
        '鉁': '',  # caractere mal codificado
        '喔': '',  # caractere mal codificado
        '檮': '',  # caractere mal codificado
        '槶': '',  # caractere mal codificado
        '槨': '',  # caractere mal codificado
        
        # Caracteres problemáticos adicionais encontrados
        '檚': '',  # caractere CJK problemático
        '崙': '',  # caractere CJK problemático
        '槇': '',  # caractere CJK problemático
        '崋': '',  # caractere CJK problemático
        '敟': '',  # caractere CJK problemático
        '猬': '',  # caractere CJK problemático
        '囷': '',  # caractere CJK problemático
        'い': '',   # hiragana
        '崝': '',  # caractere CJK problemático
        '崯': '',  # caractere CJK problemático
        '憤': '',  # caractere CJK problemático
        '徏': '',  # caractere CJK problemático
        '槒': '',  # caractere CJK problemático
        '構': '',  # caractere CJK problemático
        '憖': '',  # caractere CJK problemático
        'わ': '',  # hiragana
        '嵓': '',  # caractere CJK problemático
        '挦': '',  # caractere CJK problemático
        '敒': '',  # caractere CJK problemático
        '拫': '',  # caractere CJK problemático
        '挅': '',  # caractere CJK problemático
        '憛': '',  # caractere CJK problemático
        '挜': '',  # caractere CJK problemático
        '挄': '',  # caractere CJK problemático
        
        # Caracteres problemáticos adicionais encontrados
        'コ': '',  # katakana problemático
        '挵': '',  # caractere CJK problemático
        '憦': '',  # caractere CJK problemático
        '幁': '',  # caractere CJK problemático
    }
    
    # Aplicar correções
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Corrigir sequências de emojis mal codificadas
    # Padrão para emojis que foram quebrados em múltiplos caracteres
    emoji_patterns = [
        (r'馃叞锔忦焻帮笍', '😂'),  # emoji rindo simples
    ]
    
    for pattern, replacement in emoji_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Limpeza sistemática de caracteres CJK problemáticos
    # Substituir todos os caracteres CJK por ?
    text = re.sub(r'[\u4E00-\u9FFF]', '', text)  # Caracteres CJK
    text = re.sub(r'[\uAC00-\uD7AF]', '', text)  # Hangul
    text = re.sub(r'[\u3040-\u309F]', '', text)   # Hiragana (remover)
    text = re.sub(r'[\u30A0-\u30FF]', '', text)  # Katakana
    
    # Remover todas as menções @user
    text = re.sub(r'@user\b', '', text)  # Remove @user
    text = re.sub(r'@\w+\b', '', text)   # Remove qualquer @menção
    
    # Remover todas as palavras "rt"
    text = re.sub(r'\brt\b', '', text)  # Remove "rt" como palavra completa
    
    # Remover caracteres de substituição
    text = text.replace('�', '')  # Remove caracteres de substituição
    
    # Limpar espaços extras que podem ter sobrado
    #text = re.sub(r'\s+', ' ', text)  # Múltiplos espaços para um só
    #text = text.strip()  # Remover espaços no início e fim
    
    return text

def apply_double_encoding_fix(text):
    """
    Aplica a técnica de re-encoding: encode('latin1').decode('utf-8')
    Esta técnica é muito eficaz para corrigir problemas de codificação dupla
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    try:
        # Aplicar a técnica de re-encoding
        # Isso corrige casos onde UTF-8 foi lido como Latin-1 e vice-versa
        fixed_text = text.encode('latin1').decode('utf-8')
        return fixed_text
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Se der erro, retorna o texto original
        return text
    except Exception:
        # Qualquer outro erro, retorna o texto original
        return text

def fix_encoding_issues_with_reencoding(text):
    """
    Corrige problemas de codificação usando substituições + re-encoding
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Primeiro, aplicar as correções manuais
    text = fix_encoding_issues(text)
    
    # Depois, aplicar a técnica de re-encoding
    text = apply_double_encoding_fix(text)
    
    return text

def fix_csv_encoding(input_file, output_file=None):
    """
    Corrige problemas de codificação em um arquivo CSV
    """
    if output_file is None:
        output_file = input_file.replace('.csv', '_fixed.csv')
    
    print(f"Detectando codificação do arquivo: {input_file}")
    
    # Detectar codificação
    encoding = detect_encoding(input_file)
    print(f"Codificação detectada: {encoding}")
    
    # Ler o arquivo com a codificação detectada
    try:
        df = pd.read_csv(input_file, encoding=encoding)
        print(f"Arquivo lido com sucesso. Shape: {df.shape}")
    except Exception as e:
        print(f"Erro ao ler com codificação detectada: {e}")
        # Tentar com UTF-8
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
            print("Arquivo lido com UTF-8")
        except:
            # Tentar com Latin-1
            df = pd.read_csv(input_file, encoding='latin-1')
            print("Arquivo lido com Latin-1")
    
    print("Aplicando correções de codificação...")
    
    # Aplicar correções na coluna de texto (com re-encoding)
    if 'text' in df.columns:
        df['text'] = df['text'].apply(fix_encoding_issues_with_reencoding)
        print("Correções aplicadas na coluna 'text' (incluindo re-encoding)")
    
    # Salvar arquivo corrigido
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Arquivo corrigido salvo como: {output_file}")
    
    return df

def main():
    """Função principal"""
    input_file = "ToLD-BR.csv"
    output_file = "ToLD-BR_fixed.csv"
    
    print("=== Corretor de Codificação ToLD-BR ===")
    print(f"Arquivo de entrada: {input_file}")
    print(f"Arquivo de saída: {output_file}")
    print()
    
    # Verificar se o arquivo existe
    if not Path(input_file).exists():
        print(f"ERRO: Arquivo {input_file} não encontrado!")
        return
    
    # Corrigir codificação
    try:
        df_fixed = fix_csv_encoding(input_file, output_file)
        
        print("\n=== Exemplos de correções ===")
        # Mostrar alguns exemplos de correções
        if 'text' in df_fixed.columns:
            # Procurar por linhas que tinham problemas de codificação
            problematic_rows = df_fixed[df_fixed['text'].str.contains('ê|ã|í|á|ó|é|ç|à|😄|😍|😊|😭|😅|😂', na=False)]
            
            if len(problematic_rows) > 0:
                print("Algumas linhas corrigidas:")
                for i, row in problematic_rows.head(3).iterrows():
                    print(f"Linha {i}: {row['text'][:100]}...")
            else:
                print("Nenhuma linha com problemas de codificação encontrada.")
        
        print(f"\nArquivo corrigido salvo com sucesso!")
        print(f"Total de linhas processadas: {len(df_fixed)}")
        
    except Exception as e:
        print(f"ERRO durante o processamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
