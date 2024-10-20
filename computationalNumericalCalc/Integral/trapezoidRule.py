# Trapezoid Rule

import numpy as np
import math

def trapezoidal_rule(f, a, b, N):
    """Calcula a integral de f no intervalo [a, b] usando a regra do trapézio com N subintervalos."""
    # Comprimento de cada subintervalo
    h = (b - a) / N

    # Cálculo dos valores de x
    x = np.linspace(a, b, N + 1)

    # Cálculo das somas usando a regra do trapézio
    soma_trapezio = 0.5 * (f(x[0]) + f(x[-1]))  # Adiciona as alturas das extremidades
    soma_trapezio += np.sum(f(x[1:N]))  # Adiciona as alturas dos pontos intermediários
    soma_trapezio *= h  # Multiplica pela largura dos subintervalos

    return soma_trapezio


# Exemplo de uso
f = lambda x: np.cos(-x)  # Definindo a função a ser integrada
a = 0  # Limite inferior
b = 1  # Limite superior
N = 10  # Número de subintervalos

# Calculando a integral
integral = trapezoidal_rule(f, a, b, N)

# Exibindo o resultado
print("Integral (Regra do Trapézio):", integral)

