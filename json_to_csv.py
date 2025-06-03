import json
import pandas as pd
import os
import csv

input_dir = './2-output-urls-data'
output_csv = 'aria_landmarks_consolidated.csv'

# Campos principais para extrair
selected_fields = [
    'url', 'tagName', 'role', 'top', 'left', 
    'height', 'width', 'childs_count', 'className',
    'parent_landmark', 'label', 'xpath', 'word_count'
]

all_data = []

for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        try:
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for element in data:
                    filtered_element = {}
                    for field in selected_fields:
                        value = element.get(field, None)
                        
                        # Converter dicionários/listas para JSON string
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, ensure_ascii=False)  # Preserva acentos
                        
                        # Remover quebras de linha (causam células mescladas)
                        if isinstance(value, str):
                            value = value.replace('\n', ' ').replace('\r', ' ')
                        
                        filtered_element[field] = value
                    
                    all_data.append(filtered_element)
                    
        except Exception as e:
            print(f"Erro em {filename}: {str(e)}")

if all_data:
    df = pd.DataFrame(all_data)
    
    # Remover duplicatas
    df = df.drop_duplicates()
    
    # Filtrar elementos com 'role' válido
    df = df[df['role'].notna()]
    
    # Salvar CSV
    df.to_csv(
        output_csv,
        index=False,
        encoding='utf-8-sig',  
        quoting=csv.QUOTE_ALL, 
        quotechar='"',
        escapechar='\\'
    )
    print(f"CSV gerado: {output_csv} ({len(df)} registros)")
else:
    print("Nenhum dado encontrado.")
