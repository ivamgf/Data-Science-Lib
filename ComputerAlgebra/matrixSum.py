def soma_matrizes(matriz1, matriz2):
    # Verifica se as dimensões das duas matrizes são iguais
    if len(matriz1) != len(matriz2) or len(matriz1[0]) != len(matriz2[0]):
        return "As matrizes devem ter as mesmas dimensões."

    # Inicializa a matriz resultante
    resultado = []

    # Itera sobre as linhas e colunas das matrizes
    for i in range(len(matriz1)):
        linha = []
        for j in range(len(matriz1[0])):
            linha.append(matriz1[i][j] + matriz2[i][j])
        resultado.append(linha)

    return resultado


# Exemplo de uso
matriz1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

matriz2 = [
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
]

resultado = soma_matrizes(matriz1, matriz2)
for linha in resultado:
    print(linha)
