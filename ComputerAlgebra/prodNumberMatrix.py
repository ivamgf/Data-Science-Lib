def produto_escalar_matriz(escalar, matriz):
    # Inicializa a matriz resultante
    resultado = []

    # Itera sobre as linhas e colunas da matriz
    for i in range(len(matriz)):
        linha = []
        for j in range(len(matriz[0])):
            linha.append(escalar * matriz[i][j])
        resultado.append(linha)

    return resultado


# Exemplo de uso
escalar = 3
matriz = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

resultado = produto_escalar_matriz(escalar, matriz)
for linha in resultado:
    print(linha)
