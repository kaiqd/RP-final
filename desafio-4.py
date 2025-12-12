import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- 1. Carregar os dados ---
print("Carregando o dataset Breast Cancer (Classificação Binária)...")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"Número de amostras: {X.shape[0]}")
print(f"Número de features: {X.shape[1]}")

# --- 2. Dividir em treino e teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDados de Treino: {X_train.shape[0]} amostras")
print(f"Dados de Teste: {X_test.shape[0]} amostras")

# --- 3. Instanciar e Treinar Ambos os Modelos ---
print("\n--- Treinamento dos Modelos ---")

# a) Árvore de Decisão (Modelo Simples)
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3 # Restrição Extra: Limitando a profundidade da árvore
)
print("Treinando DecisionTreeClassifier (max_depth=3)...")
dt_model.fit(X_train, y_train)

# b) XGBClassifier (Modelo Avançado)
# Aplicamos a mesma restrição de profundidade para uma comparação mais justa
xgb_model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=3, # Restrição: Limitando a profundidade das árvores do ensemble
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
print("Treinando XGBClassifier (max_depth=3)...")
xgb_model.fit(X_train, y_train)

# --- 4. Fazer Predições e Avaliar ---

# Predição
y_pred_dt = dt_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Acurácia
acuracia_dt = accuracy_score(y_test, y_pred_dt)
acuracia_xgb = accuracy_score(y_test, y_pred_xgb)

# --- 5. Imprimir o Comparativo Lado a Lado ---

print("\n--- Resultado do Duelo de Modelos (Acurácia de Teste) ---")
print("=========================================================")
print(f"| {'Modelo':<25} | {'Acurácia (max_depth=3)':<25} |")
print("=========================================================")
print(f"| {'Decision Tree Classifier':<25} | {acuracia_dt:.4f} ({acuracia_dt*100:.2f}%) |")
print(f"| {'XGBoost Classifier':<25} | {acuracia_xgb:.4f} ({acuracia_xgb*100:.2f}%) |")
print("=========================================================")

# --- 6. Conclusão do Desafio ---
print("\n--- Conclusão da Comparação ---")
if acuracia_xgb > acuracia_dt:
    diferenca = acuracia_xgb - acuracia_dt
    print(f"O XGBoost é superior à Árvore de Decisão por {diferenca:.4f} pontos percentuais de acurácia.")
    print("Isso justifica o custo computacional maior, pois a combinação de várias árvores corrigindo erros (Boosting) funciona melhor do que uma única árvore simples.")
else:
    print("Para este dataset e estas restrições, a Árvore de Decisão simples performou tão bem quanto (ou melhor que) o XGBoost. Neste caso, a simplicidade pode ser preferível.")