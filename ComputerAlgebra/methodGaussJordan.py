import numpy as np


def gauss_jordan(matriz):
    # Converte a matriz para um array NumPy
    matriz = np.array(matriz, dtype=float)

    linhas, colunas = matriz.shape

    # Itera sobre cada coluna
    for i in range(linhas):
        # Se o elemento na diagonal principal for zero, troca de linha
        if matriz[i][i] == 0:
            for j in range(i + 1, linhas):
                if matriz[j][i] != 0:
                    matriz[[i, j]] = matriz[[j, i]]  # Troca de linhas
                    break

        # Divide a linha pela diagonal principal para obter o valor 1
        matriz[i] = matriz[i] / matriz[i][i]

        # Zera os elementos acima e abaixo da diagonal principal
        for j in range(linhas):
            if j != i:
                matriz[j] = matriz[j] - matriz[j][i] * matriz[i]

    return matriz


# Exemplo de uso
matriz = [
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
]

resultado = gauss_jordan(matriz)
print("Matriz escalonada (Forma de Gauss-Jordan):")
print(resultado)
