import numpy as np
import matplotlib.pyplot as plt

# Função f(x)
def f(x):
    return 2 * x**3 - 11.7 * x**2 + 17.7 * x - 5

# Intervalo para o gráfico
x_anterior = 0
x_posterior = 4

# Gera valores de x e calcula f(x)
lista_x = np.linspace(x_anterior, x_posterior, 1001)
lista_y = f(lista_x)

# Configuração do gráfico
plt.style.use("dark_background")
plt.plot(lista_x, lista_y, label='f(x) = 2x³ - 11.7x² + 17.7x - 5')
plt.axhline(0, color='gray', lw=0.5)  # Linha horizontal y=0
plt.axvline(0, color='gray', lw=0.5)  # Linha vertical x=0
plt.title("Gráfico de f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()  # Exibe o gráfico
