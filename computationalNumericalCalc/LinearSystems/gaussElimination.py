import numpy as np

def gauss_elimination(A, B):
    # Juntando a matriz A com o vetor B para formar a matriz aumentada
    n = len(B)
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    augmented_matrix = np.hstack((A, B.reshape(-1, 1)))

    # Eliminação de Gauss
    for i in range(n):
        # Verificando o pivô e troca de linha se necessário
        if augmented_matrix[i, i] == 0:
            for j in range(i + 1, n):
                if augmented_matrix[j, i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break
            else:
                return None, "O sistema é impossível ou possível indeterminado."

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
        S[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], S[i + 1:])) / augmented_matrix[i, i]

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

S, R, erros_estimados = gauss_elimination(A, B)

if S is None:
    print(R)  # Mensagem sobre a natureza do sistema
else:
    print("Solução: ", S)
    print("Resíduos (R = A*S - B): ", R)
    print("Erros estimados: ", erros_estimados)
