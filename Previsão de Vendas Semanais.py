import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===================== 1. Carregando os dados =====================
train = pd.read_csv('train.csv')
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')

# ===================== 2. Checagem inicial =====================
print(train.head())
print(features.head())
print(stores.head())
print(train.info())
print(features.info())
print(stores.info())

# ===================== 3. Preparação =====================
# Preencher valores nulos nas variáveis de interesse (regressores)
features['CPI'].fillna(features['CPI'].median(), inplace=True)
features['Unemployment'].fillna(features['Unemployment'].median(), inplace=True)

# Merge entre train e features
df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')

# Filtro para Loja 1, Departamento 1
df = df[(df['Store'] == 1) & (df['Dept'] == 1)]

# Conversão de data
df['Date'] = pd.to_datetime(df['Date'])

# Renomear colunas para o Prophet + manter regressores
df_prophet = df[['Date', 'Weekly_Sales', 'CPI', 'Unemployment']].rename(
    columns={'Date': 'ds', 'Weekly_Sales': 'y'}
)

# ===================== 4. Visualização =====================
plt.figure(figsize=(12,6))
plt.plot(df_prophet['ds'], df_prophet['y'])
plt.title('Vendas Semanais - Loja 1, Departamento 1')
plt.xlabel('Data')
plt.ylabel('Vendas')
plt.show()

# ===================== 5. Separar treino e teste =====================
train_size = int(len(df_prophet) * 0.8)
train_df = df_prophet.iloc[:train_size]
test_df = df_prophet.iloc[train_size:]

# ===================== 6. Modelagem Prophet com regressores =====================
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.add_regressor('CPI')
model.add_regressor('Unemployment')

model.fit(train_df)

# ===================== 7. Criar futuro e previsão =====================
future = df_prophet[['ds', 'CPI', 'Unemployment']]  # inclui regressores
forecast = model.predict(future)

# ===================== 8. Visualização da previsão =====================
plt.figure(figsize=(12,6))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Real')
plt.plot(forecast['ds'], forecast['yhat'], label='Previsto')
plt.axvline(x=df_prophet['ds'].iloc[train_size], color='r', linestyle='--', label='Início Teste')
plt.legend()
plt.title('Previsão de Vendas Semanais - Loja 1, Departamento 1')
plt.show()

# ===================== 9. Avaliação =====================
y_true = test_df['y'].values
y_pred = forecast['yhat'].iloc[-len(test_df):].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ===================== 10. Exportar a previsão =====================
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('previsao_loja1_depto1.csv', index=False)
print("Previsão exportada para 'previsao_loja1_depto1.csv'")
