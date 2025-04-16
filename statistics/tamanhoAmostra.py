## Tamanho da Amostra

def calcular_tamanho_amostra(E0, N):
    n0 = 1 / (E0 ** 2)
    n = (N * n0) / (N + n0)
    return round(n)

# Exemplo de uso:
E0 = 0.03  # Erro amostral tolerável de 5%
N = 180  # Tamanho da população

tamanho_amostra = calcular_tamanho_amostra(E0, N)
print(f"Tamanho da amostra necessário: {tamanho_amostra}")
