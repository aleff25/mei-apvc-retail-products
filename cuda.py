import torch

print(torch.cuda.is_available())  # True
print(torch.cuda.current_device())  # 0
print(torch.cuda.get_device_name(0))  # Seu modelo de GPU
