import os


def validate_labels(label_dir, num_classes):
    """
    Valida os arquivos de rótulos na pasta especificada.

    Args:
        label_dir (str): Caminho para a pasta `val/labels` contendo os arquivos de rótulos.
        num_classes (int): Número de classes no dataset.

    Returns:
        None
    """
    # Caminha pelos arquivos na pasta de labels
    for filename in os.listdir(label_dir):
        # Verifica se o arquivo é um `.txt`
        if filename.endswith(".txt"):
            file_path = os.path.join(label_dir, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Valida cada linha do arquivo
            for line_number, line in enumerate(lines, start=1):
                parts = line.strip().split()

                if len(parts) != 5:
                    print(f"Erro de formatação no arquivo {filename}, linha {line_number}: {line.strip()}")
                    continue

                try:
                    # Extrai os valores
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Valida o class_id
                    if not (0 <= class_id < num_classes):
                        print(
                            f"Erro no arquivo {filename}, linha {line_number}: class_id ({class_id}) fora do intervalo válido [0, {num_classes - 1}]")

                    # Valida coordenadas normalizadas
                    if not (0 <= x_center <= 1) or not (0 <= y_center <= 1):
                        print(
                            f"Erro no arquivo {filename}, linha {line_number}: coordenadas do centro ({x_center}, {y_center}) fora do intervalo [0, 1]")

                    if not (0 <= width <= 1) or not (0 <= height <= 1):
                        print(
                            f"Erro no arquivo {filename}, linha {line_number}: dimensões de caixa ({width}, {height}) fora do intervalo [0, 1]")

                except ValueError:
                    print(f"Erro de parsing no arquivo {filename}, linha {line_number}: {line.strip()}")

    print("Validação concluída!")


# Caminho para a pasta `val/labels`
label_dir = "dataset/train/labels"

# Número de classes no dataset (ajustar de acordo com o problema)
num_classes = 200  # Exemplo: para COCO, número de classes é 200

# Executa a validação
validate_labels(label_dir, num_classes)
