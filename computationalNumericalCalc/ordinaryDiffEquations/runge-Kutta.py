import numpy as np
import math

# Definição da equação diferencial ordinária de primeira ordem
def dydx(x, y):
    return y**2 + 3

# Método de Runge-Kutta de 4ª ordem
def rungeKutta(x0, y0, x, h):
    n = int((x - x0) / h)  # Número de iterações
    y = y0
    x_values = [x0]  # Para armazenar os valores de x
    y_values = [y0]  # Para armazenar os valores de y

    for i in range(1, n + 1):
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(x0 + h, y + k3)

        # Atualiza o valor de y
        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Atualiza o valor de x
        x0 = x0 + h

        # Armazena os valores para o gráfico
        x_values.append(x0)
        y_values.append(y)

    return x_values, y_values

# Função para interpolar o valor de y(x) se necessário
def interpolate(x_values, y_values, target_x):
    return np.interp(target_x, x_values, y_values)

# Programa principal
x0 = 0    # Valor inicial de x
y0 = 3  # Valor inicial de y
x = 0.4    # Valor de x no qual queremos encontrar y
h = 0.10  # Tamanho do passo

# Obter os valores de x e y a partir do método de Runge-Kutta
x_values, y_values = rungeKutta(x0, y0, x, h)

# Valor aproximado de y(3) pelo método de Runge-Kutta
y_approx_3 = interpolate(x_values, y_values, 3)

# Exibir o valor aproximado de y(3)
print(f"Valor aproximado de y(3) (Runge-Kutta, h={h}): {y_approx_3}")
