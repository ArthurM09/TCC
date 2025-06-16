import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações de Caminhos
BASE_PATH = '/Users/arthu/Downloads/teste2'
INPUT_CSV = os.path.join(BASE_PATH, 'aria_landmarks_consolidated.csv')

# Caminhos de saída
OUTPUT_PATH_BASE = os.path.join(BASE_PATH, 'dados_preparados')
IMAGES_PATH_BASE = os.path.join(BASE_PATH, 'images')


# Listas de roles e elementos HTML5 para landmarks
ARIA_LANDMARK_ROLES = [
    'banner', 'complementary', 'contentinfo', 'form', 'main', 
    'navigation', 'region', 'search', 'article'
]
HTML5_LANDMARK_ELEMENTS = {
    'HEADER': 'banner',
    'ASIDE': 'complementary',
    'FOOTER': 'contentinfo',
    'MAIN': 'main',
    'NAV': 'navigation',
    'SECTION': 'region',
    'ARTICLE': 'article',
    'FORM': 'form'
}
LANDMARKS_REQUIRING_LABEL = ['form', 'region', 'article', 'SECTION', 'ARTICLE', 'FORM']

def get_landmark_type(row):
    
    # Identifica o tipo de landmark com base na role e tagName
    role = str(row['role']).lower()
    tag_name = str(row['tagName']).upper()
    label = str(row['label']).lower()
    
    has_accessible_name = label not in ['false', 'none', 'nan'] and label.strip() != ''

    # Verifica por role ARIA explícita
    if role in ARIA_LANDMARK_ROLES:
        if role in LANDMARKS_REQUIRING_LABEL and not has_accessible_name:
            return 'non-landmark'
        return role

    # Verifica por elemento HTML5 com semântica de landmark implícita
    if tag_name in HTML5_LANDMARK_ELEMENTS:
        if tag_name in LANDMARKS_REQUIRING_LABEL and not has_accessible_name:
            return 'non-landmark'
        return HTML5_LANDMARK_ELEMENTS[tag_name]

    # Se não for nenhum dos acima, não é um landmark
    return 'non-landmark'

def create_features(df):
    print("\nIniciando engenharia de features...")
    
    # Garante que as colunas numéricas são do tipo correto
    numeric_cols_initial = ['top', 'left', 'height', 'width', 'childs_count', 'word_count']
    for col in numeric_cols_initial:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Feature - Área do elemento
    df['area'] = df['height'] * df['width']
    
    # Feature - Proporção (aspect ratio)
    df['aspect_ratio'] = df['width'] / (df['height'] + 1e-6)

    # Feature - É uma tag de alto nível?
    top_level_tags = ['HEADER', 'FOOTER', 'MAIN', 'NAV', 'ASIDE']
    df['is_top_level_tag'] = df['tagName'].str.upper().isin(top_level_tags).astype(int)

    print("Engenharia de features concluída.")
    return df

# Carregamento e Preparação dos Dados 
print("Carregando o dataset...")
try:
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig', sep=',', engine='python', on_bad_lines='skip')
    print(f"Dataset carregado com sucesso. Dimensões: {df.shape}")
except Exception as e:
    print(f"Erro ao carregar o CSV em '{INPUT_CSV}': {e}")
    exit(1)

# Aplicar a função para criar a nova coluna target
df['landmark_type'] = df.apply(get_landmark_type, axis=1)

# Filtrar apenas os ARIA landmarks para o treinamento
df_landmarks = df[df['landmark_type'] != 'non-landmark'].copy()
print(f"\nTotal de landmarks identificados: {df_landmarks.shape[0]}")

# Criar as novas features
df_landmarks = create_features(df_landmarks)

# Agrupar classes raras
MIN_SAMPLES = 10
value_counts = df_landmarks['landmark_type'].value_counts()
to_replace = value_counts[value_counts < MIN_SAMPLES].index
df_landmarks['landmark_type_filtered'] = df_landmarks['landmark_type'].replace(to_replace, 'other_landmark')

print("\nDistribuição de classes após agrupamento:")
print(df_landmarks['landmark_type_filtered'].value_counts())


# Todas as features a serem usadas
feature_cols = [
    'top', 'left', 'height', 'width', 'childs_count', 'word_count',
    'area', 'aspect_ratio', 'is_top_level_tag'
]

# Tratar valores nulos nas colunas de features (substituindo pela média)
for col in feature_cols:
    if df_landmarks[col].isnull().sum() > 0:
        print(f"Substituindo {df_landmarks[col].isnull().sum()} nulos na coluna '{col}' pela média.")
        df_landmarks[col].fillna(df_landmarks[col].mean(), inplace=True)

# Preparar X (features) e y (target)
X = df_landmarks[feature_cols]
y_str = df_landmarks['landmark_type_filtered']

# Codificar os rótulos (y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_str)

# Salvar o mapeamento de classes
np.save(os.path.join(OUTPUT_PATH_BASE, 'label_encoder_classes.npy'), label_encoder.classes_)
print(f"\nMapeamento de classes salvo. Total de classes: {len(label_encoder.classes_)}")

# Divisão dos Dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nConjunto de treino: {X_train.shape[0]} amostras")
print(f"Conjunto de teste: {X_test.shape[0]} amostras")


# Normalizador (Scaler) para MLP e SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salvar dados para MLP (com normalização)
print("\nSalvando dados para o MLP...")
path_mlp = os.path.join(OUTPUT_PATH_BASE, 'mlp')
np.save(os.path.join(path_mlp, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(path_mlp, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(path_mlp, 'y_train.npy'), y_train)
np.save(os.path.join(path_mlp, 'y_test.npy'), y_test)

# Salvar dados para SVM (com normalização)
print("Salvando dados para o SVM...")
path_svm = os.path.join(OUTPUT_PATH_BASE, 'svm')
os.makedirs(path_svm, exist_ok=True) # Garante que a pasta exista
np.save(os.path.join(path_svm, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(path_svm, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(path_svm, 'y_train.npy'), y_train)
np.save(os.path.join(path_svm, 'y_test.npy'), y_test)

# Salvar dados para Árvore de Decisão (sem normalização)
print("Salvando dados para a Árvore de Decisão...")
path_dt = os.path.join(OUTPUT_PATH_BASE, 'decision_tree')
np.save(os.path.join(path_dt, 'X_train.npy'), X_train.values)
np.save(os.path.join(path_dt, 'X_test.npy'), X_test.values)
np.save(os.path.join(path_dt, 'y_train.npy'), y_train)
np.save(os.path.join(path_dt, 'y_test.npy'), y_test)

# Salvar os nomes das features para referência
with open(os.path.join(OUTPUT_PATH_BASE, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(feature_cols))

print("\nDados para todos os modelos salvos com sucesso!")

# Visualizações
print("\nGerando visualizações...")
plt.figure(figsize=(12, 7))
sns.countplot(y=y_str, order=y_str.value_counts().index)
plt.title('Distribuição Final das Classes de Landmark', fontsize=16)
plt.xlabel('Contagem', fontsize=12)
plt.ylabel('Tipo de Landmark', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH_BASE, 'class_distribution_final.png'))
print("Gráfico de distribuição de classes salvo.")

# Matriz de correlação das features
plt.figure(figsize=(16, 12))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Matriz de Correlação das Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH_BASE, 'feature_correlation_matrix.png'))
print("Gráfico da matriz de correlação salvo.")

print("\nProcesso de preparação de dados concluído!")