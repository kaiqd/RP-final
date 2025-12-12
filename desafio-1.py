import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Carregando os dados
print("Carregando dataset California Housing")
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
y = california_housing.target

# O preco mediano da casa (target 'y') esta em unidades de $100,000
# O MAE e o RMSE resultantes tambem estarao nessa unidade, a menos que seja feita a conversao.
# Manteremos a unidade original para a modelagem e converteremos apenas no final

# Preparando os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dados de treino: {X_train.shape[0]} amostras")
print(f"Dados de teste: {X_test.shape[0]} amostras")

# Usar XGBRegressor e treinar o modelo
print("\nTreinando o modelo XGBRegressor...")

xgb_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100, # numero de arvores
    learning_rate=0.1,
    random_state=42
    )

xgb_model.fit(X_train, y_train)

# Fazer previsoes no conjunto de teste
y_pred = xgb_model.predict(X_test)

# Mudar a metrica de avaliacao

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# imprimindo erro medio da previsao em dolares

# os valores de 'y' no dataset estao em $100,000
# para obter o erro real em dolares, multiplicamos as metricas por 100000
CONVERSAO_DOLAR = 100000

rmse_dolar = rmse * CONVERSAO_DOLAR
mae_dolar = mae * CONVERSAO_DOLAR

print("\n--- Resultados da Avaliação do Modelo ---")
print(f"RMSE (Unidades de 100k): {rmse:.4f}")
print(f"MAE (Unidades de 100k): {mae:.4f}")
print("-" * 35)
print(f"Erro Médio (RMSE) da Previsão em Dólares: **${rmse_dolar:,.2f}**")
print(f"Erro Médio Absoluto (MAE) da Previsão em Dólares: **${mae_dolar:,.2f}**")