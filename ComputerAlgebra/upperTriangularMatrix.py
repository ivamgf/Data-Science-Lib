import numpy as np


def escalonar_para_triangular_superior(matriz):
    # Converte a matriz para um array NumPy (com tipo float para divisões)
    matriz = np.array(matriz, dtype=float)
    linhas, colunas = matriz.shape

    # Itera sobre cada coluna para aplicar o escalonamento
    for i in range(linhas):
        # Verifica se o pivô é zero. Se for, troca a linha.
        if matriz[i][i] == 0:
            for j in range(i + 1, linhas):
                if matriz[j][i] != 0:
                    matriz[[i, j]] = matriz[[j, i]]  # Troca as linhas
                    break

        # Para cada linha abaixo da linha pivô, fazemos a eliminação
        for j in range(i + 1, linhas):
            fator = matriz[j][i] / matriz[i][i]  # Fator de eliminação
            matriz[j] = matriz[j] - fator * matriz[i]  # Elimina o valor abaixo do pivô

    return matriz


# Exemplo de uso
matriz = [
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
]

resultado = escalonar_para_triangular_superior(matriz)
print("Matriz escalonada (Triangular Superior):")
print(resultado)
