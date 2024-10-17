import numpy as np
import matplotlib.pyplot as plt

# Definição da função
def f(x):
    return 2 * x**3 - 11.7 * x**2 + 17.7 * x - 5

# Método da secante
def secante(a, b, iteracoes):
    x_0 = a
    x_1 = b
    for i in range(iteracoes):
        chute = x_0 - f(x_0) * (x_1 - x_0) / (f(x_1) - f(x_0))
        x_0 = x_1
        x_1 = chute
    erro_rel = (x_1 - x_0) / x_1 * 100
    return x_1, '{:.2f}%'.format(erro_rel)

# Parâmetros do método
x0 = 3
x1 = 4
num_iteracoes = 3

# Chamando o método da secante
raiz_secante, erro_relativo = secante(x0, x1, num_iteracoes)

# Exibindo a raiz e o erro relativo
print(f"A raiz encontrada pelo método da secante é: {raiz_secante:.6f}")
print(f"Erro relativo: {erro_relativo}")

# Plotando o gráfico
x_vals = np.linspace(0, 4, 1001)
y_vals = f(x_vals)

plt.style.use("dark_background")
plt.plot(x_vals, y_vals, label='f(x) = 2x³ - 11.7x² + 17.7x - 5')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.scatter(raiz_secante, f(raiz_secante), color='red', zorder=5, label=f'Raiz: {raiz_secante:.6f}')  # Marca a raiz encontrada
plt.title("Gráfico de f(x) com a raiz encontrada pelo método da secante")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()  # Exibe o gráfico
