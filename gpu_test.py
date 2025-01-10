import torch
import torchvision

print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"Nome da GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma GPU detectada'}")
print(f"Quantidade de GPUs disponíveis: {torch.cuda.device_count()}")

print(f"PyTorch versão: {torch.__version__}")
print(f"Torchvision versão: {torchvision.__version__}")
print(f"Compatibilidade CUDA: {torch.version.cuda}")
print(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
