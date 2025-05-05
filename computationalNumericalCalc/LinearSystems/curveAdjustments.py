import numpy as np
import matplotlib.pyplot as plt  # Importando a biblioteca para plotagem

def is_positive_definite(A):
    """Verifica se a matriz é definida positiva."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def cholesky_decomposition(A):
    """Realiza a decomposição de Choleski de uma matriz A."""
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Verifica se o valor dentro da raiz é positivo
                value = A[i, i] - np.sum(L[i, :j] ** 2)
                if value <= 0:
                    raise ValueError("A matriz não é definida positiva. Valor inválido encontrado.")
                L[i, j] = np.sqrt(value)
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L

def cholesky_solve(L, B):
    """Resolve o sistema L*L^T * x = B usando a decomposição de Choleski."""
    # Resolução de L*y = B (substituição direta)
    y = np.zeros_like(B)
    for i in range(len(B)):
        y[i] = (B[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Resolução de L^T*x = y (substituição para trás)
    x = np.zeros_like(B)
    for i in range(len(B)-1, -1, -1):
        x[i] = (y[i] - np.dot(L[i+1:, i], x[i+1:])) / L[i, i]

    return x

def polynomial_curve_fit(x_data, y_data, degree):
    """Ajusta uma curva polinomial aos dados (x_data, y_data) usando mínimos quadrados com Choleski."""
    n = len(x_data)
    degree += 1  # Grau do polinômio

    # Construção da matriz de Vandermonde para os coeficientes polinomiais
    A = np.vander(x_data, degree, increasing=True)

    # Matriz A^T * A e vetor A^T * y
    ATA = np.dot(A.T, A)
    ATy = np.dot(A.T, y_data)

    # Verificar se a matriz ATA é definida positiva
    if not is_positive_definite(ATA):
        # Adiciona um pequeno valor positivo à diagonal para garantir a definição positiva
        ATA += np.eye(ATA.shape[0]) * 1e-10

    # Decomposição de Choleski de ATA
    L = cholesky_decomposition(ATA)

    # Solução do sistema L * L^T * c = A^T * y
    coef = cholesky_solve(L, ATy)

    return coef

def evaluate_polynomial(coef, x):
    """Avalia um polinômio nos pontos x, dado o vetor de coeficientes."""
    return np.polyval(coef[::-1], x)  # Reverso dos coeficientes para avaliação

# Exemplo de uso
# Dados ajustados para garantir que a matriz A^T * A seja definida positiva
x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
y_data = np.array([0, 0.8, 0.9, 0.1, -0.6, -0.8, -1, -0.9, -0.4])  # Dados quadráticos perfeitos (y = x^2)
degree = 2  # Ajustar com polinômio quadrático

# Ajuste da curva
try:
    coef = polynomial_curve_fit(x_data, y_data, degree)
    print("Coeficientes do polinômio ajustado: ", coef)

    # Avaliação do polinômio ajustado
    x_test = np.linspace(0, 8, 100)  # Gerando 100 pontos entre 0 e 8
    y_fit = evaluate_polynomial(coef, x_test)

    print("Valores ajustados: ", y_fit)
    print("Valores originais: ", y_data)

    # Resíduos (diferença entre os valores ajustados e os reais)
    residuos = y_data - evaluate_polynomial(coef, x_data)
    print("Resíduos: ", residuos)

    # Plotando os resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='red', label='Dados originais', s=100)  # Pontos originais
    plt.plot(x_test, y_fit, color='blue', label='Curva ajustada', linewidth=2)  # Curva ajustada
    plt.title('Ajuste de Curva Polinomial')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()

    # Exibindo a equação do polinômio no gráfico
    equation_text = "y = " + " + ".join(f"{coef[i]:.2f}x^{i}" for i in range(len(coef)))
    plt.text(0.1, 0.9 * max(y_fit), equation_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

except ValueError as e:
    print(e)
