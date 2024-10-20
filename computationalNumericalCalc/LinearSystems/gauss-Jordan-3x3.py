import numpy as np

def gauss_jordan(A, B):
    n = len(B)
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    augmented_matrix = np.hstack((A, B.reshape(-1, 1)))

    # Gauss-Jordan: transformação em forma escalonada reduzida
    for i in range(n):
        # Escolhendo o pivô
        if augmented_matrix[i, i] == 0:
            # Verifica se é necessário trocar de linha para evitar pivô zero
            for j in range(i + 1, n):
                if augmented_matrix[j, i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break
            else:
                return None, "O sistema é impossível ou possível indeterminado."

        # Normalizando a linha para que o pivô seja 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # Zerando as outras entradas na coluna do pivô
        for j in range(n):
            if j != i:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] -= factor * augmented_matrix[i]

    # Verificando se o sistema é possível determinado ou indeterminado
    if np.all(augmented_matrix[:, :-1] == 0, axis=1).any():
        return None, "O sistema é impossível."

    # Extraindo a solução
    S = augmented_matrix[:, -1]

    # Cálculo dos resíduos R = A*S - B
    R = np.dot(A, S) - B

    # Cálculo dos erros estimados como valor absoluto dos resíduos
    erros_estimados = np.abs(R)

    return S, R, erros_estimados

# Exemplo de uso
A = [[0.02, -1.0, 1.0],
     [0.323, 0.023, 0.89],
     [0.183, 0.143, 0.005]]

B = [2, -1, 4]

S, R, erros_estimados = gauss_jordan(A, B)

if S is None:
    print(R)  # Mensagem sobre a natureza do sistema
else:
    print("Solução: ", S)
    print("Resíduos (R = A*S - B): ", R)
    print("Erros estimados: ", erros_estimados)
