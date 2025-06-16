import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pickle
import os

# Configurações de Caminhos
BASE_PATH = '/Users/arthu/Downloads/teste2'
DATA_PATH = os.path.join(BASE_PATH, 'dados_preparados', 'decision_tree')
COMMON_DATA_PATH = os.path.join(BASE_PATH, 'dados_preparados')
OUTPUT_PATH = os.path.join(BASE_PATH, 'relatorios')
IMAGES_PATH = os.path.join(BASE_PATH, 'images', 'decision_tree')
MODELS_PATH = os.path.join(BASE_PATH, 'modelos')

# Garante que os diretórios de saída existam
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# Carregamento dos Dados 
print("Carregando os dados preparados para Árvore de Decisão...")
X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_train_encoded = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_test_encoded = np.load(os.path.join(DATA_PATH, "y_test.npy"))

# Carregar nomes das features e classes
with open(os.path.join(COMMON_DATA_PATH, "feature_names.txt"), "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

label_encoder_classes = np.load(os.path.join(COMMON_DATA_PATH, "label_encoder_classes.npy"), allow_pickle=True)
class_names = label_encoder_classes.tolist()

print(f"Dados carregados: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste")
print(f"Features: {feature_names}")
print(f"Classes: {class_names}")

# Treinamento do Modelo 
print("\nTreinando o modelo de árvore de decisão...")
dt_classifier = DecisionTreeClassifier(
    max_depth=10,          # Aumentei um pouco a profundidade para capturar mais padrões
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_classifier.fit(X_train, y_train_encoded)

print("\nAvaliando o modelo no conjunto de teste...")
y_pred_encoded = dt_classifier.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Acurácia: {accuracy:.4f}")

# Gerar relatório de classificação
print("\nRelatório de classificação:")
class_report = classification_report(y_test_encoded, y_pred_encoded, target_names=class_names, zero_division=0)
print(class_report)

# Salvar o relatório em um arquivo
with open(os.path.join(OUTPUT_PATH, "dt_report.txt"), "w") as f:
    f.write(f"Acurácia: {accuracy:.4f}\n\n")
    f.write("Relatório de classificação:\n")
    f.write(class_report)
 
# Visualizar a árvore de decisão
plt.figure(figsize=(25, 15))
plot_tree(dt_classifier, 
          feature_names=feature_names, 
          class_names=class_names,
          filled=True, 
          rounded=True, 
          max_depth=4) # Limitar a profundidade para melhor visualização
plt.title("Árvore de Decisão (Profundidade 4)", fontsize=20)
plt.savefig(os.path.join(IMAGES_PATH, "decision_tree.png"), dpi=300, bbox_inches="tight")
print("\nÁrvore de decisão salva.")

# Criar matriz de confusão
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_encoded, y_pred_encoded, labels=np.arange(len(class_names)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Árvore de Decisão Landmarks")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, "confusion_matrix.png"))
print("Matriz de confusão salva.")

# Visualizar a importância das features
feature_importance = dt_classifier.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\nImportância das features:")
print(importance_df)

plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Importância das Features - Árvore de Decisão")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, "feature_importance.png"))
print("Gráfico de importância das features salvo.")

# Salvar o modelo treinado
with open(os.path.join(MODELS_PATH, "decision_tree_model.pkl"), "wb") as f:
    pickle.dump(dt_classifier, f)
print("\nModelo salvo.")

print("\nTreinamento e avaliação da árvore de decisão concluídos com sucesso!")