import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pickle
import time
import os

# Configurações de Caminhos
BASE_PATH = '/Users/arthu/Downloads/teste2'
DATA_PATH = os.path.join(BASE_PATH, 'dados_preparados', 'mlp')
COMMON_DATA_PATH = os.path.join(BASE_PATH, 'dados_preparados')
OUTPUT_PATH = os.path.join(BASE_PATH, 'relatorios')
IMAGES_PATH = os.path.join(BASE_PATH, 'images', 'mlp')
MODELS_PATH = os.path.join(BASE_PATH, 'modelos')

# Garante que os diretórios de saída existam
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

print("Carregando os dados preparados para MLP...")
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

print("\nTreinando o modelo MLP (Rede Neural)...")
start_time = time.time()

mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="constant",
    learning_rate_init=0.001,
    max_iter=300,
    shuffle=True,
    random_state=42,
    tol=1e-4,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

mlp_classifier.fit(X_train, y_train_encoded)

end_time = time.time()
print(f"Treinamento concluído em {end_time - start_time:.2f} segundos.")

print("\nAvaliando o modelo no conjunto de teste...")
y_pred_encoded = mlp_classifier.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Acurácia: {accuracy:.4f}")

# Gerar relatório de classificação
print("\nRelatório de classificação:")
class_report = classification_report(y_test_encoded, y_pred_encoded, zero_division=0, target_names=class_names)
print(class_report)

# Salvar o relatório em um arquivo
with open(os.path.join(OUTPUT_PATH, "mlp_report.txt"), "w") as f:
    f.write(f"Acurácia: {accuracy:.4f}\n\n")
    f.write("Relatório de classificação:\n")
    f.write(class_report)

# Criar matriz de confusão
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_encoded, y_pred_encoded, labels=np.arange(len(class_names)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - MLP Landmarks")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, "mlp_confusion_matrix.png"))
print("\nMatriz de confusão salva.")

# Salvar o modelo treinado
with open(os.path.join(MODELS_PATH, "mlp_model.pkl"), "wb") as f:
    pickle.dump(mlp_classifier, f)
print("Modelo MLP salvo.")

print("\nTreinamento e avaliação do MLP para Landmarks concluídos com sucesso!")