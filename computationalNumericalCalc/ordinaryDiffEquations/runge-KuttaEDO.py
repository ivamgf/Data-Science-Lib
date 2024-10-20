from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import math

# Método de Runge-Kutta de 4ª ordem para resolver EDOs
def ODE_RKA(f, t0, tf, s0, h):
    t = np.arange(t0, tf + h, h)  # Gera a discretização do tempo
    s = np.zeros(len(t))  # Inicializa a solução s(t)
    s[0] = s0  # Condição inicial
    for i in range(0, len(t) - 1):
        f1 = h * f(t[i], s[i])
        f2 = h * f(t[i] + h / 2, s[i] + f1 / 2)
        f3 = h * f(t[i] + h / 2, s[i] + f2 / 2)
        f4 = h * f(t[i] + h, s[i] + f3)
        s[i + 1] = s[i] + (f1 + 2 * (f2 + f3) + f4) / 6
    return t, s

# Definindo a EDO (exemplo: y' = cos(t) + sin(t))
def dydt(t, y):
    return np.cos(t) + np.sin(t)

# Função exata para a EDO acima (esta função depende da EDO real)
def exact_solution(t):
    # A solução exata aqui é uma aproximação.
    return np.sin(t) - np.cos(t)

# Função para interpolar o valor de y em t=3, se necessário
def interpolate(t_values, s_values, target_t):
    return np.interp(target_t, t_values, s_values)

# Parâmetros de entrada
t0 = 0.2  # Tempo inicial
tf = 3    # Tempo final
s0 = 0.2    # Condição inicial
h = 0.3   # Tamanho do passo

# Solução numérica usando Runge-Kutta de 4ª ordem
t_values, s_values = ODE_RKA(dydt, t0, tf, s0, h)

# Solução exata
exact_values = exact_solution(t_values)

# Valor específico de y(3) aproximado pela solução numérica
y_approx_3 = interpolate(t_values, s_values, 3)

# Valor específico de y(3) da solução exata
y_exact_3 = exact_solution(3)

# Exibir os valores aproximado e exato para y(3)
print(f"Valor aproximado de y(3) (Runge-Kutta): {y_approx_3}")
print(f"Valor exato de y(3): {y_exact_3}")

# Parte gráfica comparando a solução exata com o Runge-Kutta
plt.figure(figsize=(8, 6))

# Gráfico da solução numérica (Runge-Kutta)
plt.plot(t_values, s_values, label="Runge-Kutta 4ª Ordem", marker='o', linestyle='-', color='cyan')

# Gráfico da solução exata
plt.plot(t_values, exact_values, label="Solução Exata", linestyle='--', color='yellow')

# Adicionando título e rótulos
plt.title("Comparação: Runge-Kutta 4ª Ordem vs Solução Exata")
plt.xlabel("t")
plt.ylabel("s(t)")

# Adiciona uma grade e legenda
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Exibe o gráfico
plt.show()
