import numpy as np


def determinante(matriz):
    # Verifica se a matriz é quadrada
    if len(matriz) != len(matriz[0]):
        return "A matriz deve ser quadrada."

    # Caso base: se for uma matriz 2x2, aplicamos a fórmula direta
    if len(matriz) == 2:
        return matriz[0][0] * matriz[1][1] - matriz[0][1] * matriz[1][0]

    # Expansão por cofatores para matrizes de dimensões maiores
    det = 0
    for c in range(len(matriz)):
        # Obtemos o menor complementar (submatriz)
        submatriz = [linha[:c] + linha[c + 1:] for linha in matriz[1:]]
        # Aplicamos a regra de sinais (-1)^(linha+coluna) e chamamos a função recursivamente
        det += ((-1) ** c) * matriz[0][c] * determinante(submatriz)

    return det


# Exemplo de uso para uma matriz 4x4
matriz_4x4 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

resultado_4x4 = determinante(matriz_4x4)
print("Determinante da matriz 4x4:", resultado_4x4)
