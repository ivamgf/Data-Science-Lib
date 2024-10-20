import numpy as np


def lagrange_interpolacao(x_vals, y_vals, x):
    """
    Calcula o valor interpolado no ponto x utilizando o polinômio de Lagrange.

    x_vals: lista ou array com os valores de x dados
    y_vals: lista ou array com os valores de y correspondentes
    x: ponto onde se deseja calcular a interpolação

    Retorna o valor interpolado no ponto x.
    """
    n = len(x_vals)
    interpolacao = 0

    for i in range(n):
        termo_Li = 1
        for j in range(n):
            if j != i:
                termo_Li *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        interpolacao += termo_Li * y_vals[i]

    return interpolacao


# Exemplo de uso: pontos dados para interpolação
x_vals = np.array([1, 2, 3, 4], dtype=float)  # Valores de x conhecidos
y_vals = np.array([1, 4, 9, 16], dtype=float)  # Valores de y conhecidos (neste caso, y = x^2)

# Ponto onde queremos calcular a interpolação
x_ponto = 2.5

# Calculando a interpolação de Lagrange
resultado_interpolado = lagrange_interpolacao(x_vals, y_vals, x_ponto)

print(f"O valor interpolado no ponto x = {x_ponto} é: {resultado_interpolado:.6f}")
