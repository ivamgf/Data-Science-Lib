import numpy as np

# Substituição retroativa (para matrizes triangulares superiores)
def SubRet(U, bs):
    n = bs.size
    xs = np.zeros(n)

    for i in range(n-1, -1, -1):  # Começa da última linha até a primeira
        xs[i] = (bs[i] - U[i, i+1:] @ xs[i+1:]) / U[i, i]

    return xs

# Eliminação de Gauss
def ElimGauss(inA, inbs):
    A = np.copy(inA)
    bs = np.copy(inbs)
    n = bs.size
    for j in range(n-1):
        for i in range(j+1, n):
            # O multiplicador deve ser A[i, j] / A[j, j]
            m = A[i, j] / A[j, j]
            # A operação de eliminação é feita a partir da linha j
            A[i, j:] -= m * A[j, j:]
            bs[i] -= m * bs[j]
    # Resolução via substituição retroativa
    xs = SubRet(A, bs)
    return xs

# Teste com um sistema de equações
A = np.array([[2, -1, 1], [3, 3, 9], [3, 3, 5]], dtype=float)
bs = np.array([2, -1, 4], dtype=float)

# Resolver o sistema
xs = ElimGauss(A, bs)

# Exibir a solução
print("Solução do sistema:")
print(xs)
