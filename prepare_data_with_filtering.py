import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
import csv

print("Carregando o dataset com separador vírgula e codificação utf-8-sig...")
df = pd.read_csv('/Users/arthu/GitHub/TCC/aria_landmarks_consolidated.csv', 
                 encoding='utf-8-sig',
                 sep=',',
                 engine='python',
                 on_bad_lines='skip')

print(f"Dataset carregado com sucesso. Dimensões: {df.shape}")
print(f"Colunas encontradas: {df.columns.tolist()}")

# Verificar valores únicos na coluna 'role' (target)
print("\nDistribuição de classes na coluna 'role':")
role_counts = df['role'].value_counts()
print(role_counts)

# Tratar valores nulos na coluna 'role'
print("\nVerificando nulos na coluna 'role':")
null_roles = df['role'].isnull().sum()
print(f"Valores nulos encontrados: {null_roles}")

# Preencher nulos com 'missing'
df['role'] = df['role'].fillna('missing')

# Filtrar classes raras (com menos de 5 exemplos)
MIN_SAMPLES = 5
role_counts = df['role'].value_counts()  # sem nulos
rare_classes = role_counts[role_counts < MIN_SAMPLES].index.tolist()

if 'missing' not in rare_classes:
    rare_classes.append('missing')

print(f"\nClasses com menos de {MIN_SAMPLES} exemplos (serão agrupadas como 'other'):")
for cls in rare_classes:
    print(f"- {cls}: {role_counts.get(cls, 0)} exemplos")

# Agrupar classes raras em uma categoria 'other'
print("\nAgrupando classes raras...")
df['role_filtered'] = df['role'].apply(lambda x: 'other' if x in rare_classes else x)

# Verificar a nova distribuição
print("\nNova distribuição de classes após agrupamento:")
new_role_counts = df['role_filtered'].value_counts()
print(new_role_counts)

# Identificar colunas numéricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nColunas numéricas identificadas ({len(numeric_cols)}):")
for col in numeric_cols:
    print(f"- {col}")

# Converte word_count para numérico se não estiver nas colunas numéricas
if 'word_count' not in numeric_cols and 'word_count' in df.columns:
    try:
        print("\nTentando converter word_count para numérico...")
        
        df['word_count'] = df['word_count'].astype(str).str.replace('.', '') # Remove pontos que separam milhares
        df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce')  # Converte para float

        print("Conversão concluída.")
        
        # Atualiza lista de colunas numéricas
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    except Exception as e:
        print(f"Erro ao converter word_count: {e}")

# Remove a coluna target se estiver nas colunas numéricas
if 'role' in numeric_cols:
    numeric_cols.remove('role')
if 'role_filtered' in numeric_cols:
    numeric_cols.remove('role_filtered')

# Verifica valores nulos
print("\nVerificando valores nulos nas colunas numéricas:")
null_counts = df[numeric_cols].isnull().sum()
print(null_counts)

# Substitui valores nulos pela média
print("\nSubstituindo valores nulos pela média...")
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Prepara X (features) e y (target)
X = df[numeric_cols]
y = df['role_filtered']

# Dividindo em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nConjunto de treino: {X_train.shape[0]} amostras")
print(f"Conjunto de teste: {X_test.shape[0]} amostras")

# Normaliza os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salvando os dados preparados
np.save('/Users/arthu/GitHub/TCC/dados_preparados/X_train.npy', X_train_scaled)
np.save('/Users/arthu/GitHub/TCC/dados_preparados/X_test.npy', X_test_scaled)
np.save('/Users/arthu/GitHub/TCC/dados_preparados/y_train.npy', y_train.to_numpy())
np.save('/Users/arthu/GitHub/TCC/dados_preparados/y_test.npy', y_test.to_numpy())

# Salvar os nomes das features e classes para referência
with open('/Users/arthu/GitHub/TCC/dados_preparados/feature_names.txt', 'w') as f:
    f.write('\n'.join(numeric_cols))
    
with open('/Users/arthu/GitHub/TCC/dados_preparados/class_names.txt', 'w') as f:
    f.write('\n'.join(y.unique()))

print("\nDados preparados e salvos com sucesso!")

# Mostrar estatísticas descritivas das features
print("\nEstatísticas descritivas das features:")
print(X.describe())

# Visualizar a distribuição das classes
plt.figure(figsize=(12, 6))
new_role_counts.plot(kind='bar')
plt.title('Distribuição das Classes (role_filtered)')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/arthu/GitHub/TCC/images/class_distribution.png')
print("\nGráfico de distribuição de classes salvo como 'class_distribution.png'")
