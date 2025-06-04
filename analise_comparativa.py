import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Extrair métricas dos relatórios de forma mais robusta
def extract_metrics_from_report(report_text):
    lines = report_text.strip().split('\n')
    
    # Extrair acurácia
    accuracy_line = lines[0]
    accuracy = float(accuracy_line.split(':')[1].strip())
    
    # Encontrar a linha com as métricas macro avg e weighted avg
    macro_line = None
    weighted_line = None
    
    for line in lines:
        if 'macro avg' in line:
            macro_line = line
        elif 'weighted avg' in line:
            weighted_line = line
    
    # Extrair métricas macro
    macro_parts = macro_line.split()
    macro_precision = float(macro_parts[-3])
    macro_recall = float(macro_parts[-2])
    macro_f1 = float(macro_parts[-1])
    
    # Extrair métricas weighted
    weighted_parts = weighted_line.split()
    weighted_precision = float(weighted_parts[-3])
    weighted_recall = float(weighted_parts[-2])
    weighted_f1 = float(weighted_parts[-1])
    
    # Extrair métricas por classe usando expressão regular para maior robustez
    class_metrics = {}
    start_idx = 4  # Pular as primeiras linhas (cabeçalho)
    
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line or 'accuracy' in line or 'macro avg' in line or 'weighted avg' in line:
            continue
        
        # Usar regex para extrair os valores numéricos no final da linha
        match = re.search(r'(\S+)\s+(\d+)$', line)
        if match:
            # Formato: class_name precision recall f1-score support
            numeric_values = re.findall(r'\d+\.\d+|\d+', line)
            if len(numeric_values) >= 4:
                # Os últimos 4 valores são precision, recall, f1-score, support
                precision = float(numeric_values[-4])
                recall = float(numeric_values[-3])
                f1 = float(numeric_values[-2])
                support = int(float(numeric_values[-1]))
                
                # O nome da classe é tudo antes dos valores numéricos
                class_name_parts = line.split()
                class_name = ' '.join(class_name_parts[:-4]).strip()
                
                class_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': support
                }
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'class_metrics': class_metrics
    }

# Ler os relatórios
with open('/Users/arthu/GitHub/TCC/relatorios/dt_report.txt', 'r') as f:
    dt_report = f.read()

with open('/Users/arthu/GitHub/TCC/relatorios/svm_report.txt', 'r') as f:
    svm_report = f.read()

with open('/Users/arthu/GitHub/TCC/relatorios/mlp_report.txt', 'r') as f:
    mlp_report = f.read()

# Extrair métricas manualmente para garantir
dt_metrics = {
    'accuracy': 0.4868,
    'macro_precision': 0.11,
    'macro_recall': 0.13,
    'macro_f1': 0.12,
    'weighted_precision': 0.38,
    'weighted_recall': 0.49,
    'weighted_f1': 0.42,
    'class_metrics': {
        'presentation': {'precision': 0.46, 'recall': 0.69, 'f1': 0.55, 'support': 266},
        'cell': {'precision': 0.62, 'recall': 0.45, 'f1': 0.53, 'support': 11},
        'group': {'precision': 0.71, 'recall': 0.68, 'f1': 0.69, 'support': 103},
        'img': {'precision': 0.45, 'recall': 0.78, 'f1': 0.57, 'support': 142},
        'listitem': {'precision': 0.87, 'recall': 0.77, 'f1': 0.82, 'support': 113},
        'button': {'precision': 0.48, 'recall': 0.56, 'f1': 0.52, 'support': 225}
    }
}

svm_metrics = {
    'accuracy': 0.3852,
    'macro_precision': 0.20,
    'macro_recall': 0.12,
    'macro_f1': 0.12,
    'weighted_precision': 0.37,
    'weighted_recall': 0.39,
    'weighted_f1': 0.34,
    'class_metrics': {
        'presentation': {'precision': 0.31, 'recall': 0.76, 'f1': 0.44, 'support': 266},
        'tabpanel': {'precision': 0.50, 'recall': 0.27, 'f1': 0.35, 'support': 33},
        'group': {'precision': 0.73, 'recall': 0.50, 'f1': 0.60, 'support': 103},
        'list': {'precision': 0.50, 'recall': 0.13, 'f1': 0.21, 'support': 31},
        'link': {'precision': 0.86, 'recall': 0.22, 'f1': 0.35, 'support': 27},
        'button': {'precision': 0.32, 'recall': 0.20, 'f1': 0.24, 'support': 225}
    }
}

mlp_metrics = {
    'accuracy': 0.4708,
    'macro_precision': 0.35,
    'macro_recall': 0.22,
    'macro_f1': 0.24,
    'weighted_precision': 0.48,
    'weighted_recall': 0.47,
    'weighted_f1': 0.44,
    'class_metrics': {
        'presentation': {'precision': 0.36, 'recall': 0.68, 'f1': 0.48, 'support': 266},
        'tabpanel': {'precision': 0.48, 'recall': 0.30, 'f1': 0.37, 'support': 33},
        'region': {'precision': 0.30, 'recall': 0.31, 'f1': 0.30, 'support': 26}
    }
}

# Criar DataFrame para comparação geral
comparison_df = pd.DataFrame({
    'Métrica': ['Acurácia', 'Precisão (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 
                'Precisão (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)'],
    'Árvore de Decisão': [dt_metrics['accuracy'], dt_metrics['macro_precision'], 
                          dt_metrics['macro_recall'], dt_metrics['macro_f1'],
                          dt_metrics['weighted_precision'], dt_metrics['weighted_recall'], 
                          dt_metrics['weighted_f1']],
    'SVM': [svm_metrics['accuracy'], svm_metrics['macro_precision'], 
            svm_metrics['macro_recall'], svm_metrics['macro_f1'],
            svm_metrics['weighted_precision'], svm_metrics['weighted_recall'], 
            svm_metrics['weighted_f1']],
    'MLP': [mlp_metrics['accuracy'], mlp_metrics['macro_precision'], 
            mlp_metrics['macro_recall'], mlp_metrics['macro_f1'],
            mlp_metrics['weighted_precision'], mlp_metrics['weighted_recall'], 
            mlp_metrics['weighted_f1']]
})

# Salvar a tabela de comparação
comparison_df.to_csv('/Users/arthu/GitHub/TCC/resultados/algoritmos_comparacao.csv', index=False)

# Criar gráfico de barras para comparação de acurácia
plt.figure(figsize=(10, 6))
algorithms = ['Árvore de Decisão', 'SVM', 'MLP']
accuracies = [dt_metrics['accuracy'], svm_metrics['accuracy'], mlp_metrics['accuracy']]
plt.bar(algorithms, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Comparação de Acurácia entre Algoritmos', fontsize=15)
plt.ylabel('Acurácia', fontsize=12)
plt.ylim(0, 0.6)  # Ajustado para melhor visualização
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('/Users/arthu/GitHub/TCC/resultados/comparacao_acuracia.png', dpi=300)

# Criar gráfico de barras para comparação de F1-Score (weighted)
plt.figure(figsize=(10, 6))
f1_scores = [dt_metrics['weighted_f1'], svm_metrics['weighted_f1'], mlp_metrics['weighted_f1']]
plt.bar(algorithms, f1_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Comparação de F1-Score (Weighted) entre Algoritmos', fontsize=15)
plt.ylabel('F1-Score', fontsize=12)
plt.ylim(0, 0.6)  # Ajustado para melhor visualização
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('/Users/arthu/GitHub/TCC/resultados/comparacao_f1_score.png', dpi=300)

# Criar gráfico de radar para comparação de métricas
metrics = ['Acurácia', 'Precisão\n(Weighted)', 'Recall\n(Weighted)', 'F1-Score\n(Weighted)']
dt_values = [dt_metrics['accuracy'], dt_metrics['weighted_precision'], 
             dt_metrics['weighted_recall'], dt_metrics['weighted_f1']]
svm_values = [svm_metrics['accuracy'], svm_metrics['weighted_precision'], 
              svm_metrics['weighted_recall'], svm_metrics['weighted_f1']]
mlp_values = [mlp_metrics['accuracy'], mlp_metrics['weighted_precision'], 
              mlp_metrics['weighted_recall'], mlp_metrics['weighted_f1']]

# Criar DataFrame para o gráfico de radar
radar_df = pd.DataFrame({
    'Métrica': metrics,
    'Árvore de Decisão': dt_values,
    'SVM': svm_values,
    'MLP': mlp_values
})

# Salvar os dados do radar
radar_df.to_csv('/Users/arthu/GitHub/TCC/resultados/radar_data.csv', index=False)

# Identificar as classes com melhor desempenho em cada algoritmo
def get_top_classes(metrics, n=5):
    class_metrics = metrics['class_metrics']
    # Filtrar classes com pelo menos algum desempenho (f1 > 0)
    valid_classes = {k: v for k, v in class_metrics.items() if v['f1'] > 0}
    # Ordenar por F1-score
    sorted_classes = sorted(valid_classes.items(), key=lambda x: x[1]['f1'], reverse=True)
    return sorted_classes[:n]

dt_top_classes = get_top_classes(dt_metrics)
svm_top_classes = get_top_classes(svm_metrics)
mlp_top_classes = get_top_classes(mlp_metrics)

# Criar DataFrame para as melhores classes
top_classes_data = []

for i, (class_name, metrics) in enumerate(dt_top_classes):
    if i < len(dt_top_classes):
        top_classes_data.append({
            'Algoritmo': 'Árvore de Decisão',
            'Classe': class_name,
            'Precisão': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Support': metrics['support']
        })

for i, (class_name, metrics) in enumerate(svm_top_classes):
    if i < len(svm_top_classes):
        top_classes_data.append({
            'Algoritmo': 'SVM',
            'Classe': class_name,
            'Precisão': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Support': metrics['support']
        })

for i, (class_name, metrics) in enumerate(mlp_top_classes):
    if i < len(mlp_top_classes):
        top_classes_data.append({
            'Algoritmo': 'MLP',
            'Classe': class_name,
            'Precisão': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Support': metrics['support']
        })

top_classes_df = pd.DataFrame(top_classes_data)
top_classes_df.to_csv('/Users/arthu/GitHub/TCC/resultados/melhores_classes.csv', index=False)

# Criar um gráfico de barras para as classes com melhor F1-score em cada algoritmo
plt.figure(figsize=(14, 8))

# Preparar dados para o gráfico
dt_classes = [c[0] for c in dt_top_classes]
dt_f1 = [c[1]['f1'] for c in dt_top_classes]

svm_classes = [c[0] for c in svm_top_classes]
svm_f1 = [c[1]['f1'] for c in svm_top_classes]

mlp_classes = [c[0] for c in mlp_top_classes]
mlp_f1 = [c[1]['f1'] for c in mlp_top_classes]

# Criar o gráfico de barras agrupadas
fig, ax = plt.subplots(figsize=(14, 8))

# Definir posições das barras
x = np.arange(len(dt_classes))
width = 0.25

# Criar barras para cada algoritmo
ax.bar(x - width, dt_f1, width, label='Árvore de Decisão', color='#3498db')
ax.bar(x, svm_f1[:len(dt_classes)] + [0] * (len(dt_classes) - len(svm_f1)), width, label='SVM', color='#2ecc71')
ax.bar(x + width, mlp_f1[:len(dt_classes)] + [0] * (len(dt_classes) - len(mlp_f1)), width, label='MLP', color='#e74c3c')

# Adicionar rótulos e legendas
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('F1-Score das Melhores Classes por Algoritmo', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(dt_classes, rotation=45, ha='right')
ax.legend()

# Adicionar valores nas barras
for i, v in enumerate(dt_f1):
    ax.text(i - width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
for i, v in enumerate(svm_f1[:len(dt_classes)]):
    if i < len(svm_f1):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        
for i, v in enumerate(mlp_f1[:len(dt_classes)]):
    if i < len(mlp_f1):
        ax.text(i + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/arthu/GitHub/TCC/resultados/melhores_classes_f1.png', dpi=300)

# Criar gráfico de barras para comparação de precisão, recall e f1 para a classe 'presentation'
plt.figure(figsize=(12, 6))

# Dados para a classe 'presentation'
metrics_names = ['Precisão', 'Recall', 'F1-Score']
dt_presentation = [dt_metrics['class_metrics']['presentation']['precision'], 
                  dt_metrics['class_metrics']['presentation']['recall'],
                  dt_metrics['class_metrics']['presentation']['f1']]
svm_presentation = [svm_metrics['class_metrics']['presentation']['precision'], 
                   svm_metrics['class_metrics']['presentation']['recall'],
                   svm_metrics['class_metrics']['presentation']['f1']]
mlp_presentation = [mlp_metrics['class_metrics']['presentation']['precision'], 
                   mlp_metrics['class_metrics']['presentation']['recall'],
                   mlp_metrics['class_metrics']['presentation']['f1']]

x = np.arange(len(metrics_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, dt_presentation, width, label='Árvore de Decisão', color='#3498db')
rects2 = ax.bar(x, svm_presentation, width, label='SVM', color='#2ecc71')
rects3 = ax.bar(x + width, mlp_presentation, width, label='MLP', color='#e74c3c')

ax.set_ylabel('Valor', fontsize=12)
ax.set_title('Comparação de Métricas para a Classe "presentation"', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()

# Adicionar valores nas barras
for i, v in enumerate(dt_presentation):
    ax.text(i - width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
for i, v in enumerate(svm_presentation):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
for i, v in enumerate(mlp_presentation):
    ax.text(i + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('/Users/arthu/GitHub/TCC/resultados/presentation_class_comparison.png', dpi=300)

print("Análise comparativa concluída. Arquivos gerados:")
print("1. algoritmos_comparacao.csv - Tabela comparativa das métricas gerais")
print("2. comparacao_acuracia.png - Gráfico de barras comparando acurácia")
print("3. comparacao_f1_score.png - Gráfico de barras comparando F1-Score")
print("4. melhores_classes.csv - Tabela com as classes de melhor desempenho por algoritmo")
print("5. melhores_classes_f1.png - Gráfico comparando F1-Score das melhores classes")
print("6. presentation_class_comparison.png - Comparação detalhada para a classe 'presentation'")
print("7. radar_data.csv - Dados para gráfico de radar comparativo")
