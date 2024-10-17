import numpy as np
import matplotlib.pyplot as plt

# Definição da função
def f(x):
    return 2 * x**3 - 11.7 * x**2 + 17.7 * x - 5

# Método da bisseção
def bissecao(a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        print("A função deve ter sinais opostos em a e b.")
        return None

    for i in range(max_iter):
        c = (a + b) / 2  # Ponto médio
        if abs(f(c)) < tol:  # Verifica se a raiz foi encontrada
            print(f"Raiz encontrada: {c:.6f} em {i + 1} iterações.")
            return c
        elif f(a) * f(c) < 0:  # A raiz está no intervalo [a, c]
            b = c
        else:  # A raiz está no intervalo [c, b]
            a = c

    print("Máximo de iterações alcançado.")
    return (a + b) / 2

# Parâmetros
a = 3  # Limite inferior
b = 4  # Limite superior
tolerancia = 1e-6
max_iteracoes = 100

# Chamando o método da bisseção
raiz = bissecao(a, b, tolerancia, max_iteracoes)

# Plotando o gráfico
if raiz is not None:
    x_vals = np.linspace(0, 4, 1001)
    y_vals = f(x_vals)

    plt.style.use("dark_background")
    plt.plot(x_vals, y_vals, label='f(x) = 2x³ - 11.7x² + 17.7x - 5')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.scatter(raiz, f(raiz), color='red', zorder=5, label=f'Raiz: {raiz:.6f}')  # Marca a raiz encontrada
    plt.title("Gráfico de f(x) com a raiz encontrada pelo método da bisseção")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()  # Exibe o gráfico
