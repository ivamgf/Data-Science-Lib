def transposta_matriz(matriz):
    # Inicializa a matriz transposta com zeros
    transposta = [[0 for _ in range(len(matriz))] for _ in range(len(matriz[0]))]

    # Preenche a matriz transposta
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            transposta[j][i] = matriz[i][j]

    return transposta


# Exemplo de uso
matriz = [
    [1, 2, 3],
    [4, 5, 6]
]

resultado = transposta_matriz(matriz)
for linha in resultado:
    print(linha)
