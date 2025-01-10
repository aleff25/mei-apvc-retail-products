import os
import json
from tqdm import tqdm


def convert_coco_to_yolo(json_path, output_path, images_path):
    # Abre o arquivo JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Mapeia IDs de categorias para índices de classe no formato YOLO
    category_id_to_class = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}

    # Cria um dicionário para armazenar anotações por imagem
    annotations = {img['id']: [] for img in coco_data['images']}

    # Associa cada anotação ao ID da imagem correspondente
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        bbox = ann['bbox']
        category_id = ann['category_id']

        # Converte o formato COCO (x, y, largura, altura) para YOLO (x_centro, y_centro, largura, altura normalizados)
        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2

        # Normaliza as coordenadas pelo tamanho da imagem
        image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
        img_width = image_info['width']
        img_height = image_info['height']

        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height

        # Armazena a anotação
        class_id = category_id_to_class[category_id]
        annotations[image_id].append(f"{class_id} {x_center} {y_center} {w} {h}")

    # Cria a pasta de saída se não existir
    os.makedirs(output_path, exist_ok=True)

    # Cria arquivos .txt para cada imagem
    for img in tqdm(coco_data['images'], desc="Converting COCO to YOLO format"):
        image_id = img['id']
        file_name = os.path.splitext(img['file_name'])[0]
        annotation_lines = annotations[image_id]

        label_txt_path = os.path.join(output_path, f"{file_name}.txt")
        with open(label_txt_path, 'w') as f:
            f.write("\n".join(annotation_lines))

    print(f"Conversão concluída! Anotações YOLO salvas em: {output_path}")


# Configurações (caminhos do JSON, pasta de saída para os txt e imagens)
json_path = "C:/Users/aleff/PycharmProjects/APVC/dataset/instances_val2019.json"
output_path = "C:/Users/aleff/PycharmProjects/APVC/dataset/val/labels"
images_path = "C:/Users/aleff/PycharmProjects/APVC/dataset/val/images"

convert_coco_to_yolo(json_path, output_path, images_path)
