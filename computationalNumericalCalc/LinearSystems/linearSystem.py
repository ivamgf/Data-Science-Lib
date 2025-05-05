import numpy as np

# Substituição sucessiva (para matrizes triangulares inferiores)
def SubSuc(L, bs):
    n = bs.size
    xs = np.zeros(n)

    for i in range(n):
        xs[i] = (bs[i] - L[i, :i] @ xs[:i]) / L[i, i]

    return xs

# Substituição retroativa (para matrizes triangulares superiores)
def SubRet(U, bs):
    n = bs.size
    xs = np.zeros(n)

    for i in range(n-1, -1, -1):  # Começa da última linha até a primeira
        xs[i] = (bs[i] - U[i, i+1:] @ xs[i+1:]) / U[i, i]

    return xs

# Função para criar um teste com matriz A e vetor bs
def criarTeste(n, val):
    A = np.arange(val, val + n*n).reshape(n, n)
    A = np.sqrt(A)
    bs = (A[0, :] ** 2.1)
    return A, bs

# Função para testar a solução dos métodos e compará-la com o np.linalg.solve
def solucaoTeste(Metodo, A, bs):
    print("Solução pelo método:", Metodo.__name__)
    xs = Metodo(A, bs)
    print(xs)
    print("Solução usando np.linalg.solve:")
    xs_linalg = np.linalg.solve(A, bs)
    print(xs_linalg)

# Criar o sistema de teste 4x4
A, bs = criarTeste(4, 21)

# Resolver o sistema usando SubSuc com uma matriz triangular inferior
L = np.tril(A)
solucaoTeste(SubSuc, L, bs)
print(" ")

# Resolver o sistema usando SubRet com uma matriz triangular superior
U = np.triu(A)
solucaoTeste(SubRet, U, bs)
