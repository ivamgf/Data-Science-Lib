def determinante_3x3(matriz):
    if len(matriz) != 3 or len(matriz[0]) != 3:
        return "A matriz deve ser 3x3."

    # Aplicando a fórmula para o determinante de 3x3
    a = matriz[0][0]
    b = matriz[0][1]
    c = matriz[0][2]
    d = matriz[1][0]
    e = matriz[1][1]
    f = matriz[1][2]
    g = matriz[2][0]
    h = matriz[2][1]
    i = matriz[2][2]

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# Exemplo de uso
matriz_3x3 = [
    [6, 1, 1],
    [4, -2, 5],
    [2, 8, 7]
]

resultado_3x3 = determinante_3x3(matriz_3x3)
print("Determinante da matriz 3x3:", resultado_3x3)
