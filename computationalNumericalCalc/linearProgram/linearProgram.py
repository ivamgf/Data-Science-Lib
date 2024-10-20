import numpy as np
from scipy.optimize import linprog

# Coeficientes da função objetivo (max z = c1*x1 + c2*x2 + c3*x3)
c = [-1000, -500, -1500]  # Coeficientes da função objetivo (negativos para maximização)

# Matriz de coeficientes das restrições
A = [
    [1, 1, 1],  # x1 + x2 + x3 <= 10
    [2, 3, 1],  # 2x1 + 3x2 + x3 <= 20
    [0, 2, 4],  # 2x2 + 4x3 <= 10
]

# Lado direito das restrições
b = [10, 20, 10]

# Restrições de não-negatividade (x >= 0)
bounds = [(0, None)] * 3  # x1, x2, x3 >= 0

# Resolvendo o problema de programação linear
result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# Verificando se a solução foi encontrada
if result.success:
    print("Solução encontrada:")
    print(f"Valores das variáveis: x1 = {result.x[0]}, x2 = {result.x[1]}, x3 = {result.x[2]}")
    print(f"Valor máximo da função objetivo: {-result.fun}")  # Negar porque minimizamos -c
else:
    print("Solução não foi encontrada.")
