import subprocess
import sys
import torch

def reinstall_torch_with_cuda():
    print("\n[INFO] PyTorch atual não possui suporte a GPU. Reinstalando versão com CUDA 12.1...\n")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "--index-url", "https://download.pytorch.org/whl/cu121"]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("\n[OK] PyTorch atualizado com suporte a GPU!")
    else:
        print("\n[ERRO] Falha ao reinstalar PyTorch com CUDA. Tente manualmente:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

def main():
    try:
        print(f"Versão atual do PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda

        print(f"CUDA disponível: {cuda_available}")
        print(f"Versão CUDA detectada: {cuda_version}")

        if not cuda_available or cuda_version is None:
            reinstall_torch_with_cuda()
        else:
            print("\n[OK] PyTorch já possui suporte a GPU.")
    except Exception as e:
        print(f"[ERRO] Falha ao verificar PyTorch: {e}")
        reinstall_torch_with_cuda()

if __name__ == "__main__":
    main()
