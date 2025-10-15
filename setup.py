"""
Script de setup automatizado para facilitar a instalação
"""
import subprocess
import sys
import os

def print_header(text):
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60)

def run_command(command, description):
    """Executa um comando e mostra o progresso"""
    print(f"\n{description}...")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✓ {description} - Concluído!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erro: {e}")
        return False

def main():
    print_header("SETUP DO DETECTOR DE TOXICIDADE")
    
    print("\nEste script irá:")
    print("1. Verificar a instalação do Python")
    print("2. Instalar as dependências necessárias")
    print("3. Verificar os datasets")
    print("4. Preparar o ambiente")
    
    input("\nPressione ENTER para continuar...")
    
    # 1. Verificar Python
    print_header("1. VERIFICANDO PYTHON")
    print(f"✓ Python {sys.version}")
    
    # 2. Instalar dependências
    print_header("2. INSTALANDO DEPENDÊNCIAS")
    print("Isso pode demorar alguns minutos...")
    
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Atualizando pip"
    ):
        return
    
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Instalando dependências"
    ):
        return
    
    # 3. Verificar datasets
    print_header("3. VERIFICANDO DATASETS")
    
    dataset_path = os.path.join("model_training", "data", "raw", "ToLD-BR_fixed.csv")
    if os.path.exists(dataset_path):
        print(f"✓ Dataset encontrado: {dataset_path}")
    else:
        print(f"✗ Dataset não encontrado: {dataset_path}")
        print("  Por favor, certifique-se de que o arquivo existe.")
    
    # 4. Criar diretórios
    print_header("4. PREPARANDO AMBIENTE")
    
    os.makedirs(os.path.join("model_training", "data", "processed"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("✓ Diretórios criados")
    
    # Conclusão
    print_header("SETUP CONCLUÍDO!")
    
    print("\n✅ Tudo pronto! Para usar o aplicativo, execute:")
    print(f"\n   {sys.executable} app.py")
    print("\n" + "=" * 60)
    
    input("\nPressione ENTER para sair...")

if __name__ == "__main__":
    main()


