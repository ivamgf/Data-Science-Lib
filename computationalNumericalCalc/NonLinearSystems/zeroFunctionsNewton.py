import numpy as np
import matplotlib.pyplot as plt

# Definição da função
def f(x):
    return 2 * x**3 - 11.7 * x**2 + 17.7 * x - 5

# Definição da derivada da função
def df(x):
    return 6 * x**2 - 23.4 * x + 17.7

# Método de Newton-Raphson
def newton(chute, iteracoes):
    raiz = chute
    for i in range(iteracoes):
        raiz = raiz - f(raiz) / df(raiz)
    return raiz

# Parâmetros para o método
chute_inicial = 3
num_iteracoes = 3
raiz_nova = newton(chute_inicial, num_iteracoes)

# Exibindo a raiz encontrada
print(f"A raiz encontrada pelo método de Newton-Raphson é: {raiz_nova:.6f}")

# Plotando o gráfico
x_vals = np.linspace(0, 4, 1001)
y_vals = f(x_vals)

plt.style.use("dark_background")
plt.plot(x_vals, y_vals, label='f(x) = 2x³ - 11.7x² + 17.7x - 5')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.scatter(raiz_nova, f(raiz_nova), color='red', zorder=5, label=f'Raiz: {raiz_nova:.6f}')  # Marca a raiz encontrada
plt.title("Gráfico de f(x) com a raiz encontrada pelo método de Newton-Raphson")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()  # Exibe o gráfico
