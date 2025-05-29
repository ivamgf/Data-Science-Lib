# Predict com modelo LSTM salvo

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# Função para criar as janelas de entrada com look_back
def criar_dados(dataset, look_back):
    X = []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
    return np.array(X)

# Carregar modelo, scaler e look_back
model = load_model('modelo_lstm.h5')
scaler = joblib.load('scaler.pkl')

with open('look_back.txt', 'r') as f:
    look_back = int(f.read().strip())

# Carregar novo CSV com dados para previsão
path_novo_csv = os.path.join('../Data/', 'novos_precos.csv')
df_novo = pd.read_csv(path_novo_csv, sep=';')

# Extrair e normalizar preços
precos_novos = df_novo['preco'].values.reshape(-1, 1)
precos_novos_norm = scaler.transform(precos_novos)

# Criar dados de entrada
X_novo = criar_dados(precos_novos_norm, look_back)

# Fazer predições
y_pred_norm = model.predict(X_novo)
y_pred = scaler.inverse_transform(y_pred_norm)

# Mostrar os últimos valores reais e previstos
precos_reais = precos_novos[look_back:]  # Alinhar com X_novo

# Visualização
plt.figure(figsize=(10, 5))
plt.plot(precos_reais, label='Real')
plt.plot(y_pred, label='Previsto')
plt.title('Predição com novo conjunto de dados')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.grid()
plt.show()
