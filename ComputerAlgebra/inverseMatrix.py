import numpy as np

def inversa_matriz(matriz):
    try:
        # Calcula a inversa da matriz usando a função do NumPy
        inversa = np.linalg.inv(matriz)
        return inversa
    except np.linalg.LinAlgError:
        return "A matriz não é inversível."

# Exemplo de uso
matriz = np.array([
    [4, 7],
    [2, 6]
])

inversa = inversa_matriz(matriz)
print("Matriz Inversa:")
print(inversa)
