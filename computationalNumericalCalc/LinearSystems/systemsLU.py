import numpy as np


def lu_decomposition(A):
    """ Realiza a decomposição LU de uma matriz A. """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Construção da matriz U (triangular superior)
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))

        # Construção da matriz L (triangular inferior)
        for k in range(i, n):
            if i == k:
                L[i, i] = 1  # Diagonal principal de L é 1
            else:
                L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]

    return L, U


def forward_substitution(L, b):
    """ Resolve Ly = b (sistema triangular inferior). """
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    return y


def backward_substitution(U, y):
    """ Resolve Ux = y (sistema triangular superior). """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


# Função principal para resolver o sistema de equações
def solve_lu(A, b):
    """ Resolve o sistema de equações lineares Ax = b usando decomposição LU. """
    det_A = np.linalg.det(A)

    if det_A == 0:
        print("A matriz é singular (determinante é 0). Verifique se o sistema é possível ou indeterminado.")
        return None

    # Realiza a decomposição LU
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x


# Coleta de dados do usuário para o sistema Ax = b
n = int(input("Digite o número de equações: "))  # Supondo um sistema n x n
A = np.zeros((n, n))
b = np.zeros(n)

print("Insira os coeficientes da matriz A:")
for i in range(n):
    for j in range(n):
        A[i, j] = float(input(f"A[{i + 1}][{j + 1}] = "))

print("Insira os valores do vetor b:")
for i in range(n):
    b[i] = float(input(f"b[{i + 1}] = "))

# Determinando o tipo de sistema
det_A = np.linalg.det(A)

if det_A == 0:
    print("\nO sistema possui uma matriz singular (determinante = 0).")
    # Podemos verificar aqui se é indeterminado ou impossível.
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack((A, b)))

    if rank_A == rank_Ab:
        print("O sistema é indeterminado (infinitas soluções).")
    else:
        print("O sistema é impossível (não há solução).")
else:
    # Resolvendo o sistema usando decomposição LU
    x = solve_lu(A, b)

    if x is not None:
        print("\nSoluções do sistema de equações:")
        for i in range(n):
            print(f"x[{i + 1}] = {x[i]:.6f}")
        print("O sistema é determinado (solução única).")

