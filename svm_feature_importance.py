import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import pickle
import time

# Carregar os dados preparados no "prepare_data_with_filtering.py"
print("Carregando os dados preparados...")
X_train = np.load('/Users/arthu/TCC2/dados_preparados/X_train.npy')
X_test = np.load('/Users/arthu/TCC2/dados_preparados/X_test.npy')
y_train = np.load('/Users/arthu/TCC2/dados_preparados/y_train.npy', allow_pickle=True)
y_test = np.load('/Users/arthu/TCC2/dados_preparados/y_test.npy', allow_pickle=True)

# Carregar nomes das features
with open("/Users/arthu/TCC2/dados_preparados/feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"Dados carregados: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste")
print(f"Features: {feature_names}")

# Carregar o modelo SVM treinado
print("\nCarregando o modelo SVM treinado...")
try:
    with open("/Users/arthu/TCC2/modelos/svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    print("Modelo SVM carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo svm_model.pkl não encontrado.")
    exit(1)

# Calcular a importância das features usando permutation importance
print("\nCalculando a importância das features usando permutation importance...")
print("Este processo pode levar algum tempo...")
start_time = time.time()

# Usar permutation_importance para calcular a importância das features
result = permutation_importance(
    svm_model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1  # Usar todos os processadores disponíveis
)

end_time = time.time()
print(f"Cálculo concluído em {end_time - start_time:.2f} segundos.")

# Organizar os resultados
importance = result.importances_mean
std = result.importances_std
indices = np.argsort(importance)[::-1]  # ordem decrescente

# DataFrame para melhor visualização
importance_df = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices],
    'Importance': importance[indices],
    'Std': std[indices]
}).reset_index(drop=True)

print("\nImportância das features (permutation importance):")
print(importance_df)

# Visualizar a importância das features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Importância das Features - SVM (Permutation Importance)')
plt.xlabel('Diminuição média na acurácia')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("/Users/arthu/TCC2/images/svm/svm_feature_importance.png", dpi=300)
print("\nGráfico de importância das features salvo como 'svm_feature_importance.png'")

# Salvar os resultados em um arquivo CSV
importance_df.to_csv("/Users/arthu/TCC2/dados_preparados/svm/svm_feature_importance.csv", index=False)
print("Resultados salvos em 'svm_feature_importance.csv'")

print("\nProcesso concluído com sucesso!")
