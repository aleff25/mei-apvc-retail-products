import json
import random

# Carregar o arquivo instances_val2019.json
with open('dataset/instances_val2019.json', 'r') as f:
    data = json.load(f)

# Extrair categorias
categories = data.get("categories", [])

# Gerar preços aleatórios para cada categoria
precos = {}
for category in categories:
    category_name = category.get("name")
    if category_name:
        precos[category_name] = round(random.uniform(0.50, 12.00), 2)

# Salvar os preços em um arquivo JSON
with open('prices.json', 'w') as f:
    json.dump(precos, f, indent=4)

print("Arquivo precos.json gerado com sucesso!")