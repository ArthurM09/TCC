import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pickle
import time

# Carregar os dados preparados no "prepare_data_with_filtering.py"
print("Carregando os dados preparados...")
X_train = np.load('/Users/arthu/GitHub/TCC/dados_preparados/X_train.npy')
X_test = np.load('/Users/arthu/GitHub/TCC/dados_preparados/X_test.npy')
y_train = np.load('/Users/arthu/GitHub/TCC/dados_preparados/y_train.npy', allow_pickle=True)
y_test = np.load('/Users/arthu/GitHub/TCC/dados_preparados/y_test.npy', allow_pickle=True)

# Carregar nomes das features e classes
with open("/Users/arthu/GitHub/TCC/dados_preparados/feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

with open("/Users/arthu/GitHub/TCC/dados_preparados/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Dados carregados: {X_train.shape[0]} amostras de treino, {X_test.shape[0]} amostras de teste")
print(f"Features: {feature_names}")
print(f"Classes: {class_names}")

# Criar e treinar o modelo SVM
print("\nTreinando o modelo SVM (pode levar algum tempo)...")
start_time = time.time()

# Usar SVC com kernel RBF e estratégia One-vs-Rest
svm_classifier = SVC(
    kernel="rbf", 
    C=1.0,          # Parâmetro de regularização
    gamma="scale",  # Coeficiente do kernel
    decision_function_shape="ovr", # Estratégia One-vs-Rest
    random_state=42,
    probability=True # Habilitar para predict_proba se necessário depois
)

# Treinar o modelo
svm_classifier.fit(X_train, y_train)

end_time = time.time()
print(f"Treinamento concluído em {end_time - start_time:.2f} segundos.")

# Avaliar o modelo no conjunto de teste
print("\nAvaliando o modelo no conjunto de teste...")
y_pred = svm_classifier.predict(X_test)

# Cálculo da acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.4f}")

# Gerar relatório de classificação
print("\nRelatório de classificação:")
class_report = classification_report(y_test, y_pred, zero_division=0)
print(class_report)

# Salvar o relatório em um arquivo
with open("/Users/arthu/GitHub/TCC/relatorios/svm_report.txt", "w") as f:
    f.write(f"Acurácia: {accuracy:.4f}\n\n")
    f.write("Relatório de classificação:\n")
    f.write(class_report)

# Criar matriz de confusão
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - SVM")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("/Users/arthu/GitHub/TCC/images/svm/svm_confusion_matrix.png")
print("Matriz de confusão salva como 'svm_confusion_matrix.png'")

# Salvar o modelo treinado
with open("/Users/arthu/GitHub/TCC/modelos/svm_model.pkl", "wb") as f:
    pickle.dump(svm_classifier, f)
print("\nModelo SVM salvo como 'svm_model.pkl'")

print("\nTreinamento e avaliação do SVM concluídos com sucesso!")
