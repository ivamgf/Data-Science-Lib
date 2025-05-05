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

# Função df(x) que calcula a derivada do polinômio
def df(x):
    return sum(i * coef * x**(i - 1) for i, coef in enumerate(reversed(coeficientes), start=1))

# Método de Newton-Raphson
def newton(chute, iteracoes):
    raiz = chute
    for i in range(iteracoes):
        raiz = raiz - f(raiz) / df(raiz)
    return raiz

# Parâmetros para o método
chute_inicial = float(input("Digite o chute inicial para a raiz: "))
num_iteracoes = int(input("Digite o número de iterações: "))
raiz_nova = newton(chute_inicial, num_iteracoes)

# Exibindo a raiz encontrada
print(f"A raiz encontrada pelo método de Newton-Raphson é: {raiz_nova:.6f}")

# Plotando o gráfico
x_vals = np.linspace(0, 4, 1001)
y_vals = f(x_vals)

plt.style.use("dark_background")
plt.plot(x_vals, y_vals, label='f(x)')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.scatter(raiz_nova, f(raiz_nova), color='red', zorder=5, label=f'Raiz: {raiz_nova:.6f}')  # Marca a raiz encontrada
plt.title("Gráfico de f(x) com a raiz encontrada pelo método de Newton-Raphson")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()  # Exibe o gráfico
