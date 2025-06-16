import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Configurações de Caminhos 
BASE_PATH = '/Users/arthu/Downloads/teste2'
REPORTS_PATH = os.path.join(BASE_PATH, 'relatorios')
RESULTS_PATH = os.path.join(BASE_PATH, 'resultados')

# Garante que o diretório de resultados exista
os.makedirs(RESULTS_PATH, exist_ok=True)

# Função para Extrair Métricas Automaticamente
def parse_classification_report(report_text):

    # Extrai acurácia, médias e métricas por classe de um relatório de classificação.
    metrics = {}
    lines = report_text.strip().split('\n')
    
    # Extrair Acurácia
    try:
        accuracy_match = re.search(r'Acurácia:\s*(\d+\.\d+)', report_text)
        if accuracy_match:
            metrics['accuracy'] = float(accuracy_match.group(1))
    except (ValueError, IndexError):
        metrics['accuracy'] = 0.0

    # Extrair Métricas por Classe e Médias
    class_metrics = {}
    for line in lines:
        line = line.strip()
        if not line or 'classification report' in line.lower():
            continue

        parts = re.split(r'\s{2,}', line)
        if len(parts) < 5:
            continue
            
        class_name = parts[0].strip()
        
        try:
            precision = float(parts[1])
            recall = float(parts[2])
            f1_score = float(parts[3])
            support = int(parts[4])

            if 'macro avg' in class_name:
                metrics['macro_precision'] = precision
                metrics['macro_recall'] = recall
                metrics['macro_f1'] = f1_score
            elif 'weighted avg' in class_name:
                metrics['weighted_precision'] = precision
                metrics['weighted_recall'] = recall
                metrics['weighted_f1'] = f1_score
            elif 'accuracy' not in class_name:
                class_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score,
                    'support': support
                }
        except (ValueError, IndexError):
            continue # Ignora linhas que não podem ser parseadas

    metrics['class_metrics'] = class_metrics
    return metrics

# Carregamento e Processamento dos Relatórios
reports_to_load = {
    'Árvore de Decisão': 'dt_report.txt',
    'SVM': 'svm_report.txt',
    'MLP': 'mlp_report.txt'
}

all_metrics = {}

print("Lendo e processando relatórios de classificação...")
for model_name, report_file in reports_to_load.items():
    try:
        with open(os.path.join(REPORTS_PATH, report_file), 'r') as f:
            report_content = f.read()
            all_metrics[model_name] = parse_classification_report(report_content)
            print(f"Métricas para '{model_name}' extraídas com sucesso.")
    except FileNotFoundError:
        print(f"AVISO: Arquivo de relatório '{report_file}' não encontrado. Pulando este modelo.")
        all_metrics[model_name] = {} 
    except Exception as e:
        print(f"Erro ao processar o relatório '{report_file}': {e}")
        all_metrics[model_name] = {}


# Tabela de Comparação Geral
print("\nGerando tabela de comparação geral...")
comparison_data = []
for model_name, metrics in all_metrics.items():
    if metrics: 
        comparison_data.append({
            'Algoritmo': model_name,
            'Acurácia': metrics.get('accuracy', 0),
            'Precisão (Macro)': metrics.get('macro_precision', 0),
            'Recall (Macro)': metrics.get('macro_recall', 0),
            'F1-Score (Macro)': metrics.get('macro_f1', 0),
            'Precisão (Weighted)': metrics.get('weighted_precision', 0),
            'Recall (Weighted)': metrics.get('weighted_recall', 0),
            'F1-Score (Weighted)': metrics.get('weighted_f1', 0)
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(os.path.join(RESULTS_PATH, 'algoritmos_comparacao.csv'), index=False)
print("Tabela 'algoritmos_comparacao.csv' salva.")

# Gráficos de Barras (Acurácia e F1-Score)
plt.style.use('seaborn-v0_8-whitegrid')

def plot_bar_chart(df, metric, title, filename):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Algoritmo', y=metric, data=df, palette='viridis')
    plt.title(title, fontsize=16)
    plt.ylabel(metric, fontsize=12)
    plt.xlabel('')
    plt.ylim(0, max(df[metric].max() * 1.2, 0.1))
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, filename), dpi=300)
    plt.close()

if not comparison_df.empty:
    plot_bar_chart(comparison_df, 'Acurácia', 'Comparação de Acurácia entre Algoritmos', 'comparacao_acuracia.png')
    print("Gráfico 'comparacao_acuracia.png' salvo.")
    plot_bar_chart(comparison_df, 'F1-Score (Weighted)', 'Comparação de F1-Score (Weighted) entre Algoritmos', 'comparacao_f1_score.png')
    print("Gráfico 'comparacao_f1_score.png' salvo.")

# Gráfico de Radar
print("Gerando gráfico de radar...")
radar_metrics = ['Acurácia', 'Precisão (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']
radar_data = {'group': radar_metrics}
for model_name, metrics in all_metrics.items():
    if metrics:
        radar_data[model_name] = [
            metrics.get('accuracy', 0),
            metrics.get('weighted_precision', 0),
            metrics.get('weighted_recall', 0),
            metrics.get('weighted_f1', 0)
        ]

radar_df = pd.DataFrame(radar_data)

# Número de variáveis
categories = list(radar_df['group'])
N = len(categories)

# Ângulos do eixo no gráfico de radar
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories, size=12)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
plt.ylim(0, 1)

# Plotar cada algoritmo
colors = ['#3498db', '#2ecc71', '#e74c3c']
for i, model_name in enumerate(reports_to_load.keys()):
    if model_name in radar_df.columns:
        values = radar_df[model_name].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, color=colors[i], alpha=0.25)

plt.title('Comparação de Métricas Gerais (Radar)', size=16, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig(os.path.join(RESULTS_PATH, 'comparacao_radar.png'), dpi=300)
plt.close()
print("Gráfico 'comparacao_radar.png' salvo.")

# Análise por Classe
print("Analisando desempenho por classe...")
all_class_metrics = []
for model_name, metrics in all_metrics.items():
    if metrics and 'class_metrics' in metrics:
        for class_name, class_data in metrics['class_metrics'].items():
            all_class_metrics.append({
                'Algoritmo': model_name,
                'Classe': class_name,
                'Precisão': class_data['precision'],
                'Recall': class_data['recall'],
                'F1-Score': class_data['f1'],
                'Support': class_data['support']
            })

if all_class_metrics:
    top_classes_df = pd.DataFrame(all_class_metrics)
    top_classes_df.to_csv(os.path.join(RESULTS_PATH, 'metricas_por_classe.csv'), index=False)
    print("Tabela 'metricas_por_classe.csv' salva.")

    # Gráfico de F1-Score por classe
    plt.figure(figsize=(14, 8))
    sns.barplot(data=top_classes_df, x='F1-Score', y='Classe', hue='Algoritmo', palette='viridis')
    plt.title('Comparação de F1-Score por Classe', fontsize=16)
    plt.xlabel('F1-Score', fontsize=12)
    plt.ylabel('Classe', fontsize=12)
    plt.legend(title='Algoritmo')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'comparacao_f1_por_classe.png'), dpi=300)
    plt.close()
    print("Gráfico 'comparacao_f1_por_classe.png' salvo.")

print("\nAnálise comparativa concluída!")