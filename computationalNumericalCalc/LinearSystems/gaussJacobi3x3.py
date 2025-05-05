import numpy as np


def gauss_jacobi(A, bs, x0=None, tol=1e-10, max_iter=100):
    """
    Resolve um sistema de equações lineares usando o método de Gauss-Jacobi.

    Parâmetros:
    A: Matriz de coeficientes (n x n)
    bs: Vetor de termos independentes (n)
    x0: Aproximação inicial (opcional). Se não fornecida, usa vetor de zeros.
    tol: Tolerância para a diferença entre iterações consecutivas.
    max_iter: Número máximo de iterações.

    Retorna:
    xs: Solução aproximada após a convergência ou o número máximo de iterações.
    """
    n = len(bs)
    if x0 is None:
        x0 = np.zeros(n)

    xs = np.copy(x0)
    iter_count = 0

    for k in range(max_iter):
        x_new = np.copy(xs)

        # Iteração de Gauss-Jacobi
        for i in range(n):
            sum_others = np.dot(A[i, :i], xs[:i]) + np.dot(A[i, i + 1:], xs[i + 1:])
            x_new[i] = (bs[i] - sum_others) / A[i, i]

        # Checa convergência
        if np.linalg.norm(x_new - xs, ord=np.inf) < tol:
            print(f"Convergiu em {k + 1} iterações.")
            return x_new

        xs = x_new
        iter_count += 1

    print("Máximo de iterações alcançado.")
    return xs


# Exemplo de uso para sistema 3x3
A = np.array([[10, -1, 2],
              [-1, 11, -1],
              [2, -1, 10]], dtype=float)

bs = np.array([6, 25, -11], dtype=float)
x0 = np.zeros(3)  # Aproximação inicial (opcional)

# Resolvendo o sistema
solucao = gauss_jacobi(A, bs, x0)

# Exibindo a solução
print("\nSolução do sistema:")
for i, x in enumerate(solucao):
    print(f"x{i + 1} = {x:.6f}")
