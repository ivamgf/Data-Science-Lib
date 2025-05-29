# Regression Algorithm - Treinamento e Salvamento

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Caminho para o CSV
path_csv = os.path.join('../Data/', 'precos_acoes.csv')

# Carregar dados
df = pd.read_csv(path_csv, sep=';')
precos = df['preco'].values.reshape(-1, 1)

# Normalizar dados
scaler = MinMaxScaler(feature_range=(0, 1))
precos_norm = scaler.fit_transform(precos)

# Função para criar dados com look_back
def criar_dados(dataset, look_back=10):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        y.append(dataset[i+look_back])
    return np.array(X), np.array(y)

# Preparar dados
look_back = 10
X, y = criar_dados(precos_norm, look_back)

# Divisão treino/teste
tamanho_treino = int(len(X) * 0.8)
X_train, X_test = X[:tamanho_treino], X[tamanho_treino:]
y_train, y_test = y[:tamanho_treino], y[tamanho_treino:]

# Construir modelo
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar
history = model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Avaliação
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Erro
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f'MSE: {mse:.2f}')

# Visualização
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Real')
plt.plot(y_pred_inv, label='Previsto')
plt.title('Preço Real vs Previsto')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.grid()
plt.show()

# Salvar o modelo
model.save('modelo_lstm.h5')

# Salvar o scaler
joblib.dump(scaler, 'scaler.pkl')

# Opcional: salvar look_back como constante
with open('look_back.txt', 'w') as f:
    f.write(str(look_back))
