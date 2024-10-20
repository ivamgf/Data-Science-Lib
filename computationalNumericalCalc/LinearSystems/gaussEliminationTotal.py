import numpy as np


def gauss_elimination_with_total_pivoting(A, B):
    n = len(B)
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    augmented_matrix = np.hstack((A, B.reshape(-1, 1)))

    # Vetores para rastrear trocas de linhas e colunas
    row_swap = np.arange(n)
    col_swap = np.arange(n)

    # Eliminação de Gauss com pivotamento total
    for i in range(n):
        # Encontrar o maior valor absoluto na submatriz remanescente
        max_row, max_col = np.unravel_index(np.argmax(np.abs(augmented_matrix[i:, i:n])), (n - i, n - i))
        max_row += i
        max_col += i

        # Verificar se o pivô é zero
        if augmented_matrix[max_row, max_col] == 0:
            return None, "O sistema é impossível ou possível indeterminado."

        # Trocar linhas
        if max_row != i:
            augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
            row_swap[[i, max_row]] = row_swap[[max_row, i]]  # Rastreia a troca de linhas

        # Trocar colunas
        if max_col != i:
            augmented_matrix[:, [i, max_col]] = augmented_matrix[:, [max_col, i]]
            col_swap[[i, max_col]] = col_swap[[max_col, i]]  # Rastreia a troca de colunas

        # Escalonamento
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # Verificando se o sistema é possível determinado ou indeterminado
    if np.all(augmented_matrix[:, :-1] == 0, axis=1).any():
        return None, "O sistema é impossível."

    # Substituição para trás
    S = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i + 1 < n:  # Verifica se há elementos a serem multiplicados
            S[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], S[i + 1:])) / augmented_matrix[i, i]
        else:
            S[i] = augmented_matrix[i, -1] / augmented_matrix[i, i]

    # Reverter trocas de colunas para obter a solução na ordem correta
    final_solution = np.zeros_like(S)
    for i in range(n):
        final_solution[col_swap[i]] = S[i]

    # Cálculo dos resíduos R = A*S - B
    R = np.dot(A, final_solution) - B

    # Cálculo dos erros estimados como valor absoluto dos resíduos
    erros_estimados = np.abs(R)

    return final_solution, R, erros_estimados


# Exemplo de uso
A = [[0.02, -1.0, 1.0],
     [0.323, 0.023, 0.89],
     [0.183, 0.143, 0.005]]

B = [2, -1, 4]

S, R, erros_estimados = gauss_elimination_with_total_pivoting(A, B)

if S is None:
    print(R)  # Mensagem sobre a natureza do sistema
else:
    print("Solução: ", S)
    print("Resíduos (R = A*S - B): ", R)
    print("Erros estimados: ", erros_estimados)
