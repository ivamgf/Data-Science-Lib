import numpy as np

# Exemplo de uso com NumPy
matriz_4x4 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

det = np.linalg.det(matriz_4x4)
print("Determinante da matriz 4x4 (usando NumPy):", det)
