import os

# Configurações para debug e uso de assertions CUDA
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Ativa assertions de dispositivo CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Impede execução assíncrona para debug

from ultralytics import YOLO


def filter_objects(annotation_dir, image_width, image_height, min_size_ratio=0.01):
    """
    Filtra objetos muito pequenos em arquivos de anotações.

    Args:
        annotation_dir (str): Caminho para a pasta contendo arquivos de anotações.
        image_width (int): Largura da imagem.
        image_height (int): Altura das imagens.
        min_size_ratio (float): Tamanho mínimo proporcional da largura/altura do bbox como proporção da imagem.

    Returns:
        None: Sobrescreve os arquivos de anotações com objetos filtrados.
    """
    print("Filtrando objetos pequenos em anotações...")
    for annotation_file in os.listdir(annotation_dir):
        file_path = os.path.join(annotation_dir, annotation_file)
        if annotation_file.endswith('.txt'):
            filtered_lines = []
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    cls, x, y, w, h = map(float, parts)
                    if w * image_width > min_size_ratio * image_width and h * image_height > min_size_ratio * image_height:
                        filtered_lines.append(line)

            # Remover arquivos sem anotações
            if len(filtered_lines) == 0:
                os.remove(file_path)  # Delete arquivos vazios
                print(f"Removido {file_path} (sem anotações após filtragem)")
            else:
                with open(file_path, 'w') as file:
                    file.writelines(filtered_lines)
    print("Filtragem concluída.")


def main():
    # Diretórios e hiperparâmetros
    data_yaml = '../dataset/data.yml'  # Caminho para o arquivo data.yaml
    train_annotation_dir = '../dataset/train/labels'  # Diretório de anotações de treino
    val_annotation_dir = '../dataset/val/labels'  # Diretório de anotações de validação
    image_width = 640  # Largura das imagens
    image_height = 640  # Altura das imagens
    min_bbox_size = 0.01  # Proporção mínima do tamanho dos bboxes em relação à imagem
    batch_size = 4  # Reduzido para evitar sobrecarga
    epochs = 50
    device = 'cuda'

    # Filtrar dataset antes de começar o treinamento
    filter_objects(train_annotation_dir, image_width, image_height, min_size_ratio=min_bbox_size)
    filter_objects(val_annotation_dir, image_width, image_height, min_size_ratio=min_bbox_size)

    # Realizar o treinamento
    model = YOLO("../dataset/yolo11n.pt")  # Modelo YOLO pré-treinado
    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_width,
            device=device,
            name='rpc-yolo-run',
            project='runs',
            workers=0,
            lr0=1e-3,
            amp=False
        )
    except Exception as e:
        print("Erro durante o treinamento:", e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
