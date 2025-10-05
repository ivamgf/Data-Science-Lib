# Cluster Algorithm com dados de arquivo CSV

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Leitura do arquivo CSV com dados financeiros
try:
    dados = pd.read_csv('../Data/clientes.csv')
except FileNotFoundError:
    print("Arquivo 'clientes.csv' não encontrado. Verifique o caminho e tente novamente.")
    exit()

# Verifica se as colunas necessárias existem
colunas_necessarias = {'Renda', 'Gastos', 'Investimentos'}
if not colunas_necessarias.issubset(dados.columns):
    print("O arquivo CSV deve conter as colunas: Renda, Gastos, Investimentos")
    exit()

# 2. Pré-processamento
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(dados[['Renda', 'Gastos', 'Investimentos']])

# 3. Metodo do cotovelo para encontrar o número ideal de clusters
inercia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dados_normalizados)
    inercia.append(kmeans.inertia_)

# 4. Plot do método do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(k_range, inercia, 'bo-', markersize=6)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.grid(True)
plt.show()

# 5. Solicita o número de clusters ao usuário
while True:
    try:
        k_escolhido = int(input("Digite o número de clusters desejado (ex: 3): "))
        if 1 < k_escolhido <= 10:
            break
        else:
            print("Digite um valor entre 2 e 10.")
    except ValueError:
        print("Por favor, digite um número inteiro válido.")

# 6. Aplicação do KMeans com o número de clusters escolhido
kmeans_final = KMeans(n_clusters=k_escolhido, random_state=42)
clusters = kmeans_final.fit_predict(dados_normalizados)
dados['Cluster'] = clusters

# 7. Plotagem dos clusters em 2D (usando Renda e Gastos para visualização)
plt.figure(figsize=(8, 5))
for i in range(k_escolhido):
    cluster_i = dados[dados['Cluster'] == i]
    plt.scatter(cluster_i['Renda'], cluster_i['Gastos'], label=f'Cluster {i}')

plt.xlabel('Renda')
plt.ylabel('Gastos')
plt.title('Clusters de Clientes (Renda vs Gastos)')
plt.legend()
plt.grid(True)
plt.show()
