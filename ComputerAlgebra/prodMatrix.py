def produto_matrizes(matriz1, matriz2):
    # Verifica se o número de colunas da primeira matriz é igual ao número de linhas da segunda matriz
    if len(matriz1[0]) != len(matriz2):
        return "O número de colunas da primeira matriz deve ser igual ao número de linhas da segunda matriz."

    # Inicializa a matriz resultante com zeros
    resultado = [[0 for _ in range(len(matriz2[0]))] for _ in range(len(matriz1))]

    # Multiplicação das matrizes
    for i in range(len(matriz1)):
        for j in range(len(matriz2[0])):
            for k in range(len(matriz2)):
                resultado[i][j] += matriz1[i][k] * matriz2[k][j]

    return resultado


# Exemplo de uso
matriz1 = [
    [1, 2, 3],
    [4, 5, 6]
]

matriz2 = [
    [7, 8],
    [9, 10],
    [11, 12]
]

resultado = produto_matrizes(matriz1, matriz2)
for linha in resultado:
    print(linha)
