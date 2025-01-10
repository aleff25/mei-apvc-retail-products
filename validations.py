import os

def validate_annotations(annotation_dir, num_classes):
    """
    Valida os arquivos de anotação para garantir a consistência no formato e nos valores.

    Args:
        annotation_dir (str): Caminho para o diretório contendo os arquivos de anotação.
        num_classes (int): Número de classes definido no arquivo data.yaml.

    Returns:
        None
    """
    print("Validando arquivos de anotação...")
    for annotation_file in os.listdir(annotation_dir):
        file_path = os.path.join(annotation_dir, annotation_file)
        if not annotation_file.endswith('.txt'):
            continue

        with open(file_path, 'r') as file:
            lines = file.readlines()

        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Arquivo {annotation_file}: Formato inválido (esperado 5 valores por linha)")
                continue

            cls, x, y, w, h = map(float, parts)
            if cls < 0 or cls >= num_classes:
                print(f"Arquivo {annotation_file}: Índice de classe inválido ({cls})")
                continue

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"Arquivo {annotation_file}: Valores fora do intervalo permitido ({line.strip()})")
                continue

            valid_lines.append(line)

        # Sobrescrevendo somente os dados válidos
        with open(file_path, 'w') as file:
            file.writelines(valid_lines)

        if len(valid_lines) == 0:
            print(f"Arquivo {annotation_file}: Removido (não contém anotações válidas)")

    print("Validação concluída.")


num_classes = 200  # Defina o número correto de classes
validate_annotations('dataset/train/labels', num_classes)
validate_annotations('dataset/val/labels', num_classes)

import yaml


def validate_data_yaml(yaml_path):
    """
    Valida um data.yaml para certificar-se de que os parâmetros estão corretos.

    Args:
        yaml_path (str): Caminho até o arquivo data.yaml.

    Returns:
        None
    """
    with open(yaml_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        print("Conteúdo de data.yaml:", data)
        assert 'train' in data and 'val' in data, "Os caminhos para train e val precisam estar definidos"
        assert 'nc' in data and isinstance(data['nc'], int), "O campo 'nc' deve ser definido como número inteiro"
        assert 'names' in data and isinstance(data['names'], list), "O campo 'names' precisa ser uma lista de classes"
        assert len(data['names']) == data['nc'], "O número de classes não bate com 'nc'"
        print("data.yaml validado com sucesso!")


validate_data_yaml('dataset/data.yml')


import torch

torch.cuda.empty_cache()

