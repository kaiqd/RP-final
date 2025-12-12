import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Carregar os dados ---
print("Carregando o dataset Wine (Vinho)...")
wine = load_wine()
X = wine.data
y = wine.target

# Verificar o número de classes (3 cultivadores)
classes = np.unique(y)
print(f"Classes (Cultivadores) presentes: {classes}")

# --- 2. Preparar os dados (Divisão Treino/Teste) ---
# Usamos 'stratify=y' para manter a proporção das 3 classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDados de Treino: {X_train.shape[0]} amostras")
print(f"Dados de Teste: {X_test.shape[0]} amostras")

# --- 3. Usar XGBClassifier e configurar para Multiclasse ---
print("\nTreinando o modelo XGBClassifier para Classificação Multiclasse...")

# Para multiclasse, objective='multi:softmax' e num_class deve ser 3
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(classes),  # Define o número de classes (3)
    n_estimators=100,        # Número de árvores
    learning_rate=0.1,
    use_label_encoder=False, # Boas práticas do XGBoost
    eval_metric='merror',    # Métrica de erro multiclasse
    random_state=42
)

xgb_model.fit(X_train, y_train)

# --- 4. Fazer previsões no conjunto de teste ---
y_pred = xgb_model.predict(X_test)

# --- 5. Gerar Matriz de Confusão e Acurácia ---
acuracia = accuracy_score(y_test, y_pred)
matriz_confusao = confusion_matrix(y_test, y_pred)

print("\n--- Resultados da Classificação ---")
print(f"Acurácia do Modelo: **{acuracia:.4f}**")
print("\nMatriz de Confusão (Dados Puros):\n", matriz_confusao)

# --- 6. Visualização da Matriz de Confusão (Para Análise Detalhada) ---
# Ajuda a ver onde o modelo está confundindo uma classe com a outra.

plt.figure(figsize=(8, 6))
sns.heatmap(
    matriz_confusao,
    annot=True,              # Mostrar os valores
    fmt='d',                 # Formato inteiro
    cmap='Blues',
    xticklabels=wine.target_names, # Nomes dos cultivadores nas colunas (Predito)
    yticklabels=wine.target_names  # Nomes dos cultivadores nas linhas (Verdadeiro)
)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão - Classificação de Vinhos (XGBoost)')
plt.show()

print("\n--- Interpretação da Matriz ---")
# Os acertos estão na diagonal principal.
acertos = np.diag(matriz_confusao)
total_acertos = np.sum(acertos)
print(f"Acertos por Classe (Diagonal Principal): {acertos}")
print(f"Total de amostras: {len(y_test)}. Total de acertos: {total_acertos}")