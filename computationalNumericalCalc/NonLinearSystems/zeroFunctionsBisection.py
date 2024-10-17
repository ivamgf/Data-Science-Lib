import numpy as np
import matplotlib.pyplot as plt

# Solicita os coeficientes ao usuário
coeficientes = []
graus = [6, 5, 4, 3, 2, 1, 0]  # Graus correspondentes dos coeficientes (x^6, x^5, x^4, x^3, x^2, x, constante)

for grau in graus:
    coef = float(input(f"Digite o coeficiente para x^{grau}: "))
    coeficientes.append(coef)

# Função f(x) que calcula o valor do polinômio
def f(x):
    return sum(coef * x**i for i, coef in enumerate(reversed(coeficientes)))

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
a = float(input("Digite o limite inferior (a): "))  # Limite inferior
b = float(input("Digite o limite superior (b): "))  # Limite superior
tolerancia = float(input("Digite a tolerância: "))  # Tolerância
max_iteracoes = int(input("Digite o número máximo de iterações: "))  # Máximo de iterações

# Chamando o método da bisseção
raiz = bissecao(a, b, tolerancia, max_iteracoes)

# Plotando o gráfico
if raiz is not None:
    x_vals = np.linspace(0, 4, 1001)
    y_vals = f(x_vals)

    plt.style.use("dark_background")
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.scatter(raiz, f(raiz), color='red', zorder=5, label=f'Raiz: {raiz:.6f}')  # Marca a raiz encontrada
    plt.title("Gráfico de f(x) com a raiz encontrada pelo método da bisseção")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()  # Exibe o gráfico
