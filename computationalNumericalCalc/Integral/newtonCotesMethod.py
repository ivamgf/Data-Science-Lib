# Newton - Cotes method
# Rule of rectangles

import numpy as np
import math

# Definindo a função a ser integrada
f = lambda x: math.e - x

# Limites de integração
a = 0
b = 1

# Número de subintervalos
N = 10

# Comprimento de cada subintervalo
h = (b - a) / N

# Cálculo das posições médias para a regra dos retângulos
x_med = np.linspace(a + h/2, b - h/2, N)

# Soma da área dos retângulos
soma_retangulo = np.sum(f(x_med) * h)

# Exibindo o resultado
print("Integral:", soma_retangulo)
