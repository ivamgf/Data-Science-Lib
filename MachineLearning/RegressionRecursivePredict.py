import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

look_back = 10
dias_futuros = 30  # Quantidade de dias a prever

# Carregar modelo e scaler
model = load_model('modelo_lstm.h5')
scaler = joblib.load('scaler.pkl')

# Carregar dados
dados = pd.read_csv('../Data/precos_acoes.csv', sep=';')

# Pega os preços e converte em array 2D
precos = dados['preco'].values.reshape(-1, 1)

# Checa se tem dados suficientes para criar a janela inicial
if len(precos) < look_back:
    raise ValueError(f"Tem que ter pelo menos {look_back} registros para criar a janela inicial.")

# Normaliza os preços carregados
precos_norm = scaler.transform(precos)

# Usa os últimos look_back dias para a janela inicial
entrada = precos_norm[-look_back:].tolist()

previsoes_norm = []

for _ in range(dias_futuros):
    X_input = np.array(entrada[-look_back:]).reshape(1, look_back, 1)
    y_pred = model.predict(X_input, verbose=0)
    previsoes_norm.append(y_pred[0])
    entrada.append(y_pred[0])  # atualiza a janela

# Reverte normalização para o formato original
previsoes = scaler.inverse_transform(previsoes_norm)

# Mostrar gráfico das previsões
plt.figure(figsize=(10, 5))
plt.plot(previsoes, label='Previsões Futuras')
plt.title('Previsão Recursiva dos Preços')
plt.xlabel('Dias Fututos')
plt.ylabel('Preço Previsto')
plt.legend()
plt.grid()
plt.show()

# Salvar previsões
df_previsoes = pd.DataFrame(previsoes, columns=['preco_previsto'])
df_previsoes.to_csv('../Data/previsoes_futuras.csv', index=False)
print("Previsões salvas em '../Data/previsoes_futuras.csv'")
