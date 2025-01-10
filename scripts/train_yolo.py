# scripts/treinar_yolo.py

import os
import torch
from ultralytics import YOLO


def main():
    """
    Exemplo de treinamento utilizando YOLO (Ultralytics).
    Ajuste hiperparâmetros e caminhos conforme necessário.
    """

    # Escolha seu modelo base. Você pode usar:
    # 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    # Para segmentação, pode usar 'yolov8n-seg.pt', etc.
    model = YOLO('../models/yolov10n.pt')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Treinando com o dispositivo: {device}")

    # Treinando o modelo
    model.train(
        data="../dataset/data.yml",  # caminho do data.yaml
        epochs=50,  # quantidade de épocas (ajuste conforme necessário)
        batch=4,  # tamanho do batch
        imgsz=640,  # tamanho das imagens para o treinamento
        name='rpc-yolo-run',  # nome do experimento (pasta de resultados)
        project='runs',  # pasta onde salvar os resultados
        lr0=1e-3,  # taxa de aprendizado inicial
        device='cpu',  # para usar GPU. Se tiver CPU somente, pode ser device='cpu'
        amp=False,
        mosaic=True
    )

    # Ao final do treino, o melhor modelo costuma estar em: runs/rpc-yolo-run/weights/best.pt
    # Você pode copiá-lo para a pasta models se quiser
    best_model_path = 'runs/rpc-yolo-run/weights/best.pt'
    if os.path.exists(best_model_path):
        os.makedirs('models', exist_ok=True)
        os.rename(best_model_path, os.path.join('models', 'best.pt'))
        print(f"Melhor modelo salvo em: models/best.pt")
    else:
        print("Não foi possível localizar o best.pt após o treinamento. Verifique o caminho.")


if __name__ == "__main__":
    main()
