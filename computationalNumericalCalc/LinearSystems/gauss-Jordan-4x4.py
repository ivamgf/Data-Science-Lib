import numpy as np

def gauss_jordan(matrix):
    """ Resolve um sistema de equações lineares utilizando o método de Gauss-Jordan. """
    # Converte a matriz para um array numpy
    augmented_matrix = np.array(matrix, dtype=float)

    # Número de linhas
    n = augmented_matrix.shape[0]

    for i in range(n):
        # Faz a diagonal principal ser igual a 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i][i]

        # Faz todas as linhas abaixo e acima da linha i serem iguais a 0
        for j in range(n):
            if j != i:
                augmented_matrix[j] -= augmented_matrix[j][i] * augmented_matrix[i]

    return augmented_matrix  # Retorna a matriz aumentada reduzida

# Coleta de dados do usuário
coefficients = []
for i in range(4):
    print(f"Equação {i + 1}:")
    a = float(input("Coeficiente de x: "))
    b = float(input("Coeficiente de y: "))
    c = float(input("Coeficiente de z: "))
    d = float(input("Coeficiente de w: "))
    e = float(input("Termo independente: "))
    coefficients.append([a, b, c, d, e])

# Resolvendo o sistema de equações
reduced_matrix = gauss_jordan(coefficients)

# Exibindo os resultados
print("\nSoluções do sistema de equações:")
for i in range(4):
    print(f"x_{i+1} = {reduced_matrix[i][-1]:.6f}")

# Discussão do sistema
def discussao_sistema(reduced_matrix):
    """ Discute a natureza do sistema de equações. """
    n = reduced_matrix.shape[0]  # Número de equações
    rank = np.linalg.matrix_rank(reduced_matrix[:, :-1])  # Rank da matriz dos coeficientes
    augmented_rank = np.linalg.matrix_rank(reduced_matrix)  # Rank da matriz aumentada

    if rank < n:
        if augmented_rank == rank:
            print("O sistema é **impossível** (não tem solução).")
        else:
            print("O sistema é **indeterminado** (infinitas soluções).")
    elif rank == n:
        print("O sistema é **determinado** (existe uma única solução).")
    else:
        print("O sistema não se encaixa nas condições padrão.")

# Chama a função de discussão
discussao_sistema(reduced_matrix)
