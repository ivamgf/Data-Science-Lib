import numpy as np
from scipy.optimize import linprog

# Coeficientes da função objetivo (minimizar -c, então usamos -c)
c = [280, 620, 0]  # Coeficientes da função objetivo (maximizar z = 2x1 + 3x2 - 4x3)

# Matriz de coeficientes das restrições
A = [
    [0.75, 0.6, 0],  # x1 + x2 + 3x3 <= 15
    [1, 1, 1],  # x1 + 2x2 - x3 <= 20
]

# Lado direito das restrições
b = [200, 300]

# Restrições de não-negatividade (x >= 0)
bounds = [(0, None)] * 3  # x1, x2, x3 >= 0

# Resolvendo o problema de programação linear
result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# Verificando se a solução foi encontrada
if result.success:
    print("Solução encontrada:")
    print(f"Valores das variáveis: x1 = {result.x[0]}, x2 = {result.x[1]}, x3 = {result.x[2]}")
    print(f"Valor ótimo da função objetivo: {result.fun * -1}")  # Negar porque minimizamos -c
else:
    print("Solução não foi encontrada.")
