import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb 
from sklearn.metrics import accuracy_score

# --- 1. Carregar e Preparar os Dados ---
print("Carregando o dataset Breast Cancer para o Desafio 5...")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDados de Treino: {X_train.shape[0]} amostras")
print(f"Dados de Teste (Validação): {X_test.shape[0]} amostras")


# --- 2. Preparar os dados para a API Nativa (DMatrix) ---
# DMatrix é o formato de dado otimizado do XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# --- 3. Definir Parâmetros e Treinar com Early Stopping via xgb.train ---
print("\n--- Treinamento com Early Stopping (API Nativa) ---")

# Parâmetros
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',  
    'eta': 0.1,                
    'seed': 42
}

evals = [(dtrain, 'train'), (dtest, 'eval')] 

# Executar o treino nativo
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,          
    evals=evals,
    early_stopping_rounds=10,      
    verbose_eval=False             
)

# AJUSTE CHAVE: Usar bst.best_iteration em vez de bst.best_ntree_limit
num_arvores_treinadas = bst.best_iteration
print(f"Treinamento concluído. Árvores criadas/usadas: **{num_arvores_treinadas}** (Parou antes das 1000).")

# --- 4. Salvamento do Modelo em Formato JSON (Requisito) ---
NOME_ARQUIVO = "modelo_final_xgboost.json"

try:
    print(f"\nSalvando o modelo treinado em '{NOME_ARQUIVO}'...")
    # Salvamos o modelo diretamente do objeto nativo (bst)
    bst.save_model(NOME_ARQUIVO)
    print("Modelo salvo com sucesso.")
except Exception as e:
    print(f"Erro ao salvar o modelo: {e}")

# --- 5. Carregamento e Teste do Modelo Salvo ---
print(f"\nCarregando o modelo de volta para teste...")

# Criamos um novo modelo vazio (Booster)
bst_carregado = xgb.Booster(params=params) # Opcional: passa os parâmetros novamente

# Carregar os pesos e a estrutura do arquivo
bst_carregado.load_model(NOME_ARQUIVO)
print("Modelo carregado com sucesso.")

# Fazer a predição com o modelo carregado
# Usamos o iteration_range baseado no best_iteration
y_pred_probs = bst_carregado.predict(
    dtest, 
    iteration_range=(0, bst_carregado.best_iteration)
)
y_pred_carregado = (y_pred_probs > 0.5).astype(int) 

# Calcular a acurácia
acuracia_carregada = accuracy_score(y_test, y_pred_carregado)

print("\n--- Validação do Modelo Carregado ---")
print(f"Acurácia do modelo carregado (no X_test): **{acuracia_carregada:.4f}**")
