import os
import requests

# Diretório HOME definido anteriormente
HOME = os.environ.get("HOME", os.getcwd())

# Criação do diretório 'weights'
weights_dir = os.path.join(HOME, "models")
os.makedirs(weights_dir, exist_ok=True)

# URLs dos arquivos para download
urls = [
    "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
    "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
    "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
    "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
    "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt",
    "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
]

# Fazer o download de cada arquivo
for url in urls:
    file_name = os.path.join(weights_dir, os.path.basename(url))
    print(f"Baixando {file_name}...")
    response = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(response.content)

# Listar arquivos no diretório
print("\nArquivos baixados:")
for file in os.listdir(weights_dir):
    file_path = os.path.join(weights_dir, file)
    print(f"{file} - {os.path.getsize(file_path) / 1024:.2f} KB")
