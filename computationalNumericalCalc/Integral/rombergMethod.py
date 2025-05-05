# Romberg Method

import numpy as np
import math

def f(x):
    """Função a ser integrada."""
    function = math.e - x
    return function # Defina a função desejada aqui

def romberg_integration(f, a, b, tol=1e-6, max_steps=10):
    """Calcula a integral de f no intervalo [a, b] usando o método de Romberg."""
    R = np.zeros((max_steps, max_steps))

    # Inicializando R[0,0] usando a regra do trapézio
    R[0, 0] = (b - a) * (f(a) + f(b)) / 2

    for i in range(1, max_steps):
        # Calcular R[i, 0] usando a regra do trapézio com 2^i subintervalos
        h = (b - a) / (2 ** i)
        # Somando os pontos intermediários
        sum_f = sum(f(a + j * h) for j in range(1, 2 ** i, 2))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_f

        # Calcular R[i, j] usando a extrapolação de Richardson
        for j in range(1, i + 1):
            R[i, j] = (4 ** j * R[i, j - 1] - R[i - 1, j - 1]) / (4 ** j - 1)

        # Verifica a convergência
        if abs(R[i, i] - R[i - 1, i - 1]) < tol:
            return R[i, i]

    raise ValueError("O método de Romberg não convergiu.")


# Exemplo de uso
a = 0  # Limite inferior
b = 1  # Limite superior

# Calculando a integral
integral = romberg_integration(f, a, b)

# Exibindo o resultado
print("Integral (Método de Romberg):", integral)

