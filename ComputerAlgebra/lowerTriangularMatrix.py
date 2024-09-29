import numpy as np


def escalonar_para_triangular_inferior(matriz):
    # Converte a matriz para um array NumPy (com tipo float para divisões)
    matriz = np.array(matriz, dtype=float)
    linhas, colunas = matriz.shape

    # Itera sobre cada coluna
    for i in range(linhas):
        # Se o pivô for zero, troca a linha
        if matriz[i][i] == 0:
            for j in range(i + 1, linhas):
                if matriz[j][i] != 0:
                    matriz[[i, j]] = matriz[[j, i]]  # Troca de linha
                    break

        # Para cada linha acima da linha pivô, faz a eliminação
        for j in range(i):
            fator = matriz[j][i] / matriz[i][i]  # Fator de eliminação
            matriz[j] = matriz[j] - fator * matriz[i]  # Elimina o valor acima do pivô

    return matriz


# Exemplo de uso
matriz = [
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
]

resultado = escalonar_para_triangular_inferior(matriz)
print("Matriz escalonada (Triangular Inferior):")
print(resultado)
