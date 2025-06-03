import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Carregar os dados preparados no "prepare_data_with_filtering.py"
print("Carregando os dados preparados...")
X_train = np.load('/Users/arthu/TCC2/dados_preparados/X_train.npy')
X_test = np.load('/Users/arthu/TCC2/dados_preparados/X_test.npy')
y_train = np.load('/Users/arthu/TCC2/dados_preparados/y_train.npy', allow_pickle=True)
y_test = np.load('/Users/arthu/TCC2/dados_preparados/y_test.npy', allow_pickle=True)

# Carregar nomes das features e classes
with open('/Users/arthu/TCC2/dados_preparados/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

with open('/Users/arthu/TCC2/dados_preparados/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Dados carregados: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste")
print(f"Features: {feature_names}")
print(f"Classes: {class_names}")

# Criar e treinar o modelo de árvore de decisão
print("\nTreinando o modelo de árvore de decisão...")
# Iniciar com uma árvore simples para evitar overfitting
dt_classifier = DecisionTreeClassifier(
    max_depth=5,           # Profundidade máxima da árvore
    min_samples_split=10,  # Mínimo de amostras necessárias para dividir um nó
    min_samples_leaf=5,    # Mínimo de amostras necessárias em um nó folha
    random_state=42        # Para reprodutibilidade
)

# Treinar o modelo
dt_classifier.fit(X_train, y_train)

# Avaliar o modelo no conjunto de teste
print("\nAvaliando o modelo no conjunto de teste...")
y_pred = dt_classifier.predict(X_test)

# Cálculo da acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.4f}")

# Gerar relatório de classificação
print("\nRelatório de classificação:")
class_report = classification_report(y_test, y_pred)
print(class_report)

# Salvar o relatório em um arquivo
with open('/Users/arthu/TCC2/relatorios/dt_report.txt', 'w') as f:
    f.write(f"Acurácia: {accuracy:.4f}\n\n")
    f.write("Relatório de classificação:\n")
    f.write(class_report)

# Visualizar a árvore de decisão (versão simplificada)
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=feature_names, 
          class_names=class_names,
          filled=True, 
          rounded=True, 
          max_depth=3)  # Limitar a profundidade para melhor visualização
plt.savefig('/Users/arthu/TCC2/images/decision_tree/decision_tree.png', dpi=300, bbox_inches='tight')
print("\nÁrvore de decisão salva como 'decision_tree.png'")

# Criar matriz de confusão
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/Users/arthu/TCC2/images/decision_tree/confusion_matrix.png')
print("Matriz de confusão salva como 'confusion_matrix.png'")

# Calcular a importância das features
feature_importance = dt_classifier.feature_importances_
# DataFrame para melhor visualização
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("\nImportância das features:")
print(importance_df)

# Visualizar a importância das features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Importância das Features')
plt.tight_layout()
plt.savefig('/Users/arthu/TCC2/images/decision_tree/feature_importance.png')
print("Gráfico de importância das features salvo como 'feature_importance.png'")

# Salvar o modelo treinado
import pickle
with open('/Users/arthu/TCC2/modelos/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_classifier, f)
print("\nModelo salvo como 'decision_tree_model.pkl'")

print("\nTreinamento e avaliação da árvore de decisão concluídos com sucesso!")
