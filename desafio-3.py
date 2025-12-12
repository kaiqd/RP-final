import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# --- 1. Gerar Dados Sintéticos Desbalanceados ---
# O parâmetro weights=[0.99, 0.01] cria 99% de não-fraudes (classe 0)
# e apenas 1% de fraudes (classe 1).
print("Gerando dados sintéticos: 99% Classe 0 (Não Fraude), 1% Classe 1 (Fraude)...")
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.99, 0.01],
    random_state=42,
    n_features=10,        # Apenas para ter algumas features
    n_redundant=0,
    n_informative=5
)

# Contagem para confirmar o desbalanceamento
total_fraudes = np.sum(y == 1)
print(f"Total de amostras: 1000")
print(f"Total de Fraudes (Classe 1): {total_fraudes} ({total_fraudes/1000*100:.1f}%)")

# --- 2. Dividir em treino e teste (Usando stratify) ---
# 'stratify=y' é crucial para garantir que a classe 1 (fraude)
# apareça tanto no treino quanto no teste, mantendo a proporção de 1%.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDados de Treino: {X_train.shape[0]} amostras")
print(f"Dados de Teste: {X_test.shape[0]} amostras")

# --- 3. Cenário A: Treinar Modelo Sem Ajustes (Modelo Ingênuo) ---
print("\n--- Cenário A: Modelo Padrão (Sem Ajuste para Desbalanceamento) ---")

xgb_model_padrao = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_model_padrao.fit(X_train, y_train)
y_pred_padrao = xgb_model_padrao.predict(X_test)

# Métricas do Modelo Padrão
acuracia_padrao = accuracy_score(y_test, y_pred_padrao)
recall_padrao = recall_score(y_test, y_pred_padrao) # Foco na Recall da CLASSE 1

print(f"Acurácia: {acuracia_padrao:.4f} (Parece bom, mas é enganosa!)")
print(f"Recall da Fraude (Classe 1): **{recall_padrao:.4f}**")
print("Matriz de Confusão Padrão:\n", confusion_matrix(y_test, y_pred_padrao))


# --- 4. Cenário B: Treinar Modelo com Ajuste (scale_pos_weight) ---

# O ajuste é o cálculo de: (Total de Amostras Negativas) / (Total de Amostras Positivas)
# Isso informa ao XGBoost para dar N vezes mais peso às fraudes (Classe 1).
contagem_negativos = np.sum(y_train == 0) # Não-fraudes
contagem_positivos = np.sum(y_train == 1) # Fraudes

scale_pos_weight = contagem_negativos / contagem_positivos
print(f"\n--- Cenário B: Modelo Ajustado (scale_pos_weight={scale_pos_weight:.2f}) ---")

xgb_model_ajustado = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    # PARÂMETRO CHAVE: Aumenta o peso dos erros na classe minoritária
    scale_pos_weight=scale_pos_weight
)

xgb_model_ajustado.fit(X_train, y_train)
y_pred_ajustado = xgb_model_ajustado.predict(X_test)

# Métricas do Modelo Ajustado
acuracia_ajustada = accuracy_score(y_test, y_pred_ajustado)
recall_ajustada = recall_score(y_test, y_pred_ajustado) # Foco na Recall da CLASSE 1

print(f"Acurácia: {acuracia_ajustada:.4f}")
print(f"Recall da Fraude (Classe 1): **{recall_ajustada:.4f}**")
print("Matriz de Confusão Ajustada:\n", confusion_matrix(y_test, y_pred_ajustado))


# --- 5. Comparar os Resultados (Requisito) ---
print("\n--- Comparativo de Recall (Detecção de Fraudes) ---")
print(f"Recall (Modelo Padrão): {recall_padrao:.4f}")
print(f"Recall (Modelo Ajustado com scale_pos_weight): **{recall_ajustada:.4f}**")

if recall_ajustada > recall_padrao:
    print("\nSucesso! O modelo ajustado detectou mais fraudes (Recall maior).")
else:
    print("\nAtenção! O ajuste não melhorou ou piorou a detecção de fraudes.")