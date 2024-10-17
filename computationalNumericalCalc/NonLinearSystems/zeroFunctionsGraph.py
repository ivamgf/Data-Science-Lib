import numpy as np
import matplotlib.pyplot as plt

# Solicita os coeficientes ao usuário
coeficientes = []
graus = [6, 5, 4, 3, 2, 1, 0]  # Graus correspondentes dos coeficientes

for grau in graus:
    coef = float(input(f"Digite o coeficiente para x^{grau}: "))
    coeficientes.append(coef)

# Função f(x) que calcula o valor do polinômio
def f(x):
    return sum(coef * x**i for i, coef in enumerate(reversed(coeficientes)))

# Intervalo para o gráfico
x_anterior = 0
x_posterior = 4

# Gera valores de x e calcula f(x)
lista_x = np.linspace(x_anterior, x_posterior, 1001)
lista_y = f(lista_x)

# Configuração do gráfico
plt.style.use("dark_background")
plt.plot(lista_x, lista_y, label='f(x)')
plt.axhline(0, color='gray', lw=0.5)  # Linha horizontal y=0
plt.axvline(0, color='gray', lw=0.5)  # Linha vertical x=0
plt.title("Gráfico de f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()  # Exibe o gráfico
