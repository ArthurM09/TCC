import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pickle
import time

# Carregar os dados preparados no "prepare_data_for_mlp.py"
print("Carregando os dados preparados...")
X_train = np.load("/Users/arthu/TCC2/dados_preparados/mlp/X_train.npy")
X_test = np.load("/Users/arthu/TCC2/dados_preparados/mlp/X_test.npy")
y_train_encoded = np.load("/Users/arthu/TCC2/dados_preparados/mlp/y_train_encoded.npy")
y_test_encoded = np.load("/Users/arthu/TCC2/dados_preparados/mlp/y_test_encoded.npy")

# Carregar nomes das features
with open("/Users/arthu/TCC2/dados_preparados/mlp/feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

# Carregar nomes das classes originais (mapeamento do LabelEncoder)
try:
    class_names = np.load("/Users/arthu/TCC2/dados_preparados/mlp/label_encoder_classes.npy", allow_pickle=True)
    print(f"Nomes das classes carregados. Total: {len(class_names)}")
except FileNotFoundError:
    print("Erro: Arquivo label_encoder_classes.npy não encontrado. Não será possível gerar relatórios com nomes de classes.")
    class_names = None

print(f"Dados carregados: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste")
print(f"Features: {feature_names}")

# Criar e treinar o modelo MLP
print("\nTreinando o modelo MLP (Rede Neural)...")
start_time = time.time()

# Define a arquitetura do MLP
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Duas camadas ocultas com 100 e 50 neurônios
    activation="relu",             # Função de ativação ReLU
    solver="adam",                 
    alpha=0.0001,                  # Parâmetro de regularização L2
    batch_size="auto",
    learning_rate="constant",
    learning_rate_init=0.001,
    max_iter=300,                  # Aumentar número máximo de iterações
    shuffle=True,
    random_state=42,
    tol=1e-4,
    verbose=False,                 # Desativar verbose para não poluir a saída
    early_stopping=True,           # Habilitar parada antecipada
    validation_fraction=0.1,       # Usar 10% dos dados de treino para validação
    n_iter_no_change=10            # Número de iterações sem melhora para parar
)

# Treinar o modelo com os rótulos codificados
mlp_classifier.fit(X_train, y_train_encoded)

end_time = time.time()
print(f"Treinamento concluído em {end_time - start_time:.2f} segundos.")

# Avalia o modelo no conjunto de teste
print("\nAvaliando o modelo no conjunto de teste...")
y_pred_encoded = mlp_classifier.predict(X_test)

# Calcula a acurácia
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Acurácia: {accuracy:.4f}")

# Gerar relatório de classificação
print("\nRelatório de classificação:")
if class_names is not None:
    class_report = classification_report(y_test_encoded, y_pred_encoded, zero_division=0, target_names=class_names)
else:
    class_report = classification_report(y_test_encoded, y_pred_encoded, zero_division=0)
print(class_report)

# Salvar o relatório em um arquivo
with open("/Users/arthu/TCC2/relatorios/mlp_report.txt", "w") as f:
    f.write(f"Acurácia: {accuracy:.4f}\n\n")
    f.write("Relatório de classificação:\n")
    f.write(class_report)

# Criar matriz de confusão
plt.figure(figsize=(12, 10))
if class_names is not None:
    cm = confusion_matrix(y_test_encoded, y_pred_encoded, labels=range(len(class_names)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
else:
    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - MLP")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("/Users/arthu/TCC2/images/mlp/mlp_confusion_matrix.png")
print("Matriz de confusão salva como 'mlp_confusion_matrix.png'")

# Salvar o modelo treinado
with open("/Users/arthu/TCC2/modelos/mlp_model.pkl", "wb") as f:
    pickle.dump(mlp_classifier, f)
print("\nModelo MLP salvo como 'mlp_model.pkl'")

print("\nTreinamento e avaliação do MLP concluídos com sucesso!")
