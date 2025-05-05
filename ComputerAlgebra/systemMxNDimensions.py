import numpy as np

def gauss_elimination(A, b):
    # Combina a matriz A e o vetor b na matriz aumentada
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    n = len(b)

    # Escalonamento por eliminação de Gauss
    for i in range(n):
        # Pivoteamento para evitar divisão por zero
        max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # Torna o elemento diagonal principal igual a 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # Zera os elementos abaixo da diagonal
        for j in range(i + 1, n):
            augmented_matrix[j] = augmented_matrix[j] - augmented_matrix[j, i] * augmented_matrix[i]

    # Substituição reversa para encontrar as soluções
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])

    return x

def resolve_sistema(A, b):
    try:
        # Resolve o sistema linear usando a função de eliminação de Gauss
        solucao = gauss_elimination(A, b)
        return solucao
    except Exception as e:
        return f"Erro ao resolver o sistema: {e}"

# Exemplo para uma matriz 2x3 (onde há 2 equações e 2 incógnitas)
A_2x3 = np.array([[2, -1], [4, -3]])
b_2x3 = np.array([3, 9])
solucao_2x3 = resolve_sistema(A_2x3, b_2x3)
print("Solução para o sistema 2x3:", solucao_2x3)

# Exemplo para uma matriz 3x2 (3 equações e 2 incógnitas não tem solução única)
A_3x2 = np.array([[1, 2], [2, 4], [3, 6]])
b_3x2 = np.array([5, 10, 15])
solucao_3x2 = resolve_sistema(A_3x2, b_3x2)
print("Solução para o sistema 3x2:", solucao_3x2)

# Exemplo para uma matriz 3x4 (3 equações e 3 incógnitas)
A_3x4 = np.array([[1, 2, -1], [3, 1, 2], [2, -1, 1]])
b_3x4 = np.array([3, 9, 7])
solucao_3x4 = resolve_sistema(A_3x4, b_3x4)
print("Solução para o sistema 3x4:", solucao_3x4)

