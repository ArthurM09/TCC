import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import csv

print("Carregando o dataset com separador vírgula e codificação utf-8-sig...")
try:
    df = pd.read_csv('/Users/arthu/GitHub/TCC/aria_landmarks_consolidated.csv',
        encoding='utf-8-sig', 
        sep=',',               # Usar vírgula como separador
        engine='python',       # Engine mais flexível
        on_bad_lines='skip'    # Pular linhas problemáticas
    )
    print(f"Dataset carregado com sucesso. Dimensões: {df.shape}")
    print(f"Colunas encontradas: {df.columns.tolist()}")
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    print("Verifique se o arquivo CSV está correto e se o separador é realmente a vírgula.")
    exit(1)

# Verifica valores únicos na coluna 'role' (target)
if 'role' not in df.columns:
    print("Erro: A coluna 'role' não foi encontrada após a leitura. Verifique o cabeçalho do CSV.")
    print(f"Colunas lidas: {df.columns.tolist()}")
    exit(1)

print("\nDistribuição de classes na coluna 'role':")
role_counts = df['role'].value_counts()
print(role_counts)

# Filtra classes raras (com menos de 5 exemplos)
MIN_SAMPLES = 5
rare_classes = role_counts[role_counts < MIN_SAMPLES].index.tolist()
print(f"\nClasses com menos de {MIN_SAMPLES} exemplos (serão agrupadas como 'other'):")
for cls in rare_classes:
    print(f"- {cls}: {role_counts[cls]} exemplos")

# Agrupa classes raras em uma categoria 'other'
print("\nAgrupando classes raras...")
df['role_filtered'] = df['role'].apply(lambda x: 'other' if x in rare_classes else x)

# Verifica a nova distribuição
print("\nNova distribuição de classes após agrupamento:")
new_role_counts = df['role_filtered'].value_counts()
print(new_role_counts)

print("\nVerificando e tratando classes com poucos exemplos...")

# Identifica classes com menos de 2 exemplos
classes_below_2 = new_role_counts[new_role_counts < 2].index.tolist()

if classes_below_2:
    print(f"Classes com menos de 2 exemplos encontradas: {classes_below_2}")
    
    # 1. Tenta agrupar em 'other' novamente
    print("Tentando agrupar em 'other'...")
    df['role_filtered'] = df['role_filtered'].apply(
        lambda x: 'other' if x in classes_below_2 else x
    )
    
    # 2. Verifica novamente a distribuição
    updated_counts = df['role_filtered'].value_counts()
    still_problematic = updated_counts[updated_counts < 2].index.tolist()
    
    # 3. Se ainda tiver classes problemáticas, remove essas amostras
    if still_problematic:
        print(f"Ainda existem classes problemáticas após reagrupamento: {still_problematic}")
        print(f"Removendo {len(df[df['role_filtered'].isin(still_problematic)])} amostras problemáticas...")
        df = df[~df['role_filtered'].isin(still_problematic)]
    
    # Atualiza contagem final
    final_counts = df['role_filtered'].value_counts()
    print("\nDistribuição final após tratamento:")
    print(final_counts)
else:
    print("Todas as classes têm pelo menos 2 exemplos. Prosseguindo.")

# VERIFICAÇÃO FINAL ANTES DA DIVISÃO
print("\nVerificação final de classes:")
final_class_counts = df['role_filtered'].value_counts()
min_samples = final_class_counts.min()
print(f"Número mínimo de exemplos por classe: {min_samples}")

if min_samples < 2:
    problematic_classes = final_class_counts[final_class_counts < 2].index.tolist()
    print(f"ATENÇÃO: Ainda existem classes com menos de 2 exemplos: {problematic_classes}")
    print("Removendo essas amostras como medida de segurança...")
    df = df[~df['role_filtered'].isin(problematic_classes)]
    print("\nDistribuição final corrigida:")
    print(df['role_filtered'].value_counts())

# Identifica colunas numéricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nColunas numéricas identificadas ({len(numeric_cols)}):")
for col in numeric_cols:
    print(f"- {col}")

# Converte word_count para numérico se não estiver nas colunas numéricas
if 'word_count' not in numeric_cols and 'word_count' in df.columns:
    try:
        print("\nTentando converter word_count para numérico...")
        
        df['word_count'] = df['word_count'].astype(str).str.replace('.', '', regex=False) # Remover pontos que separam milhares
        df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce') # Converter para float

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
y_str = df['role_filtered']  

# Codificar os rótulos (y) para formato numérico
print("\nCodificando rótulos da coluna target (role_filtered)...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_str)

# Salvar o mapeamento de classes
np.save('/Users/arthu/GitHub/TCC/dados_preparados/mlp/label_encoder_classes.npy', label_encoder.classes_)
print(f"Mapeamento de classes salvo em 'label_encoder_classes.npy'. Total de classes: {len(label_encoder.classes_)}")

# Verificação EXTRA de segurança
unique, counts = np.unique(y_encoded, return_counts=True)
min_count = min(counts)
print(f"\nVerificação final antes da divisão - mínimo de exemplos por classe: {min_count}")

if min_count < 2:
    problematic_classes = unique[np.where(counts < 2)]
    problematic_classes_names = label_encoder.inverse_transform(problematic_classes)
    print(f"Classes problemáticas: {problematic_classes_names}")
    
    # Cria máscara para remover amostras problemáticas
    mask = ~np.isin(y_encoded, problematic_classes)
    
    # Filtra os dados
    X = X[mask]
    y_encoded = y_encoded[mask]
    
    print(f"Removidas {len(mask) - sum(mask)} amostras problemáticas")
    print(f"Novo shape dos dados: {X.shape}")

# Agora realiza a divisão
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Dividindo em conjuntos de treino e teste usando os rótulos codificados
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

print(f"\nConjunto de treino: {X_train.shape[0]} amostras")
print(f"Conjunto de teste: {X_test.shape[0]} amostras")

# Normalizar os dados (features X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salvar os dados preparados (X normalizado e y codificado)
np.save('/Users/arthu/GitHub/TCC/dados_preparados/mlp/X_train.npy', X_train_scaled)
np.save('/Users/arthu/GitHub/TCC/dados_preparados/mlp/X_test.npy', X_test_scaled)
np.save('/Users/arthu/GitHub/TCC/dados_preparados/mlp/y_train_encoded.npy', y_train_encoded)
np.save('/Users/arthu/GitHub/TCC/dados_preparados/mlp/y_test_encoded.npy', y_test_encoded)

# Salvar também os nomes das features para referência
with open('/Users/arthu/GitHub/TCC/dados_preparados/mlp/feature_names.txt', 'w') as f:
    f.write('\n'.join(numeric_cols))

print("\nDados preparados e salvos com sucesso!")

# Mostrar estatísticas descritivas das features
print("\nEstatísticas descritivas das features:")
print(X.describe())
