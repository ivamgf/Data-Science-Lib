import numpy as np

def analise_sistema(matriz_aumentada):
    # Converte a matriz aumentada para um array NumPy
    matriz_aumentada = np.array(matriz_aumentada, dtype=float)

    # Separa a matriz dos coeficientes e a matriz dos resultados
    matriz_coeficientes = matriz_aumentada[:, :-1]
    matriz_resultados = matriz_aumentada[:, -1]

    # Calcula o rank da matriz dos coeficientes e da matriz aumentada
    rank_coeficientes = np.linalg.matrix_rank(matriz_coeficientes)
    rank_aumentada = np.linalg.matrix_rank(matriz_aumentada)

    # Verifica se o sistema é determinado, indeterminado ou impossível
    if rank_coeficientes == rank_aumentada:
        if rank_coeficientes == matriz_coeficientes.shape[1]:  # Se o rank é igual ao número de variáveis
            return "O sistema é possível e determinado."
        else:
            return "O sistema é possível e indeterminado."
    else:
        return "O sistema é impossível."

# Exemplo de uso
matriz_aumentada_1 = [
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
]

matriz_aumentada_2 = [
    [1, 2, 3, 9],
    [2, 4, 6, 18],
    [3, 6, 9, 27]
]

matriz_aumentada_3 = [
    [1, 2, 1, 10],
    [2, 4, 2, 20],
    [3, 6, 3, 25]
]

print(analise_sistema(matriz_aumentada_1))  # Possível e determinado
print(analise_sistema(matriz_aumentada_2))  # Possível e indeterminado
print(analise_sistema(matriz_aumentada_3))  # Impossível
