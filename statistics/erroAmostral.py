## Erro Amostral

import math

def calcular_tamanho_amostra(E0, N):
    n0 = 1 / (E0 ** 2)
    n = (N * n0) / (N + n0)
    return round(n)

# Exemplo de uso:
E0 = 0.05  # Erro amostral tolerável de 5%
N = 10000  # Tamanho da população

tamanho_amostra = calcular_tamanho_amostra(E0, N)

def calcular_erro_amostral(n, N):
    E0 = math.sqrt((N + n) / (N * n))
    return round(E0, 4)

# Exemplo de uso:
n = tamanho_amostra  # Usa o valor calculado anteriormente
erro_amostral = calcular_erro_amostral(n, N)
print(f"Tamanho da amostra necessário: {tamanho_amostra}")
print(f"Erro amostral real obtido: {erro_amostral}")
