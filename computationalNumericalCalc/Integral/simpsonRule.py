# Simpson Rule

import numpy as np
import math

import numpy as np


def simpsons_rule(f, a, b, N):
    """Calcula a integral de f no intervalo [a, b] usando a regra de Simpson com N subintervalos."""
    # Certificando-se de que N é par
    if N % 2 == 1:
        N += 1  # Se N é ímpar, aumentar para o próximo número par

    # Comprimento de cada subintervalo
    h = (b - a) / N

    # Cálculo dos valores de x
    x = np.linspace(a, b, N + 1)

    # Cálculo das somas usando a regra de Simpson
    soma_simpson = f(x[0]) + f(x[-1])  # Adiciona as alturas das extremidades

    # Soma das alturas dos pontos intermediários
    soma_simpson += 4 * np.sum(f(x[1:N:2]))  # Soma das alturas ímpares (multiplicado por 4)
    soma_simpson += 2 * np.sum(f(x[2:N - 1:2]))  # Soma das alturas pares (multiplicado por 2)

    # Multiplica pela largura dos subintervalos
    soma_simpson *= h / 3

    return soma_simpson


# Exemplo de uso
f = lambda x: math.e-x  # Definindo a função a ser integrada
a = 0  # Limite inferior
b = 1  # Limite superior
N = 10  # Número de subintervalos (deve ser par)

# Calculando a integral
integral = simpsons_rule(f, a, b, N)

# Exibindo o resultado
print("Integral (Regra de Simpson):", integral)

