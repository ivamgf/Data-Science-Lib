def determinante_2x2(matriz):
    if len(matriz) != 2 or len(matriz[0]) != 2:
        return "A matriz deve ser 2x2."

    # Aplicando a fórmula para o determinante de 2x2
    a = matriz[0][0]
    b = matriz[0][1]
    c = matriz[1][0]
    d = matriz[1][1]

    return a * d - b * c


# Exemplo de uso
matriz_2x2 = [
    [4, 6],
    [3, 8]
]

resultado_2x2 = determinante_2x2(matriz_2x2)
print("Determinante da matriz 2x2:", resultado_2x2)
