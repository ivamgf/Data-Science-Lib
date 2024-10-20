import numpy as np


def diferencas_divididas(x_vals, y_vals):
    """
    Calcula a tabela de diferenças divididas de Newton.

    x_vals: lista ou array com os valores de x dados
    y_vals: lista ou array com os valores de y correspondentes

    Retorna a tabela de diferenças divididas.
    """
    n = len(y_vals)
    tabela = np.zeros((n, n))  # Cria uma tabela nxn de zeros
    tabela[:, 0] = y_vals  # Primeira coluna é igual aos valores de y

    # Preenche a tabela de diferenças divididas
    for j in range(1, n):
        for i in range(n - j):
            tabela[i][j] = (tabela[i + 1][j - 1] - tabela[i][j - 1]) / (x_vals[i + j] - x_vals[i])

    return tabela


def polinomio_newton(tabela_diff, x_vals, x):
    """
    Avalia o polinômio interpolador de Newton no ponto x.

    tabela_diff: tabela de diferenças divididas de Newton
    x_vals: lista ou array com os valores de x dados
    x: ponto onde se deseja calcular a interpolação

    Retorna o valor interpolado no ponto x.
    """
    n = len(x_vals)
    resultado = tabela_diff[0, 0]  # Valor inicial (primeira diferença dividida)
    termo = 1

    for i in range(1, n):
        termo *= (x - x_vals[i - 1])
        resultado += tabela_diff[0, i] * termo

    return resultado


# Exemplo de uso: pontos dados para interpolação
x_vals = np.array([1, 2, 3, 4], dtype=float)  # Valores de x conhecidos
y_vals = np.array([1, 4, 9, 16], dtype=float)  # Valores de y conhecidos (neste caso, y = x^2)

# Calcula a tabela de diferenças divididas de Newton
tabela_diff = diferencas_divididas(x_vals, y_vals)

# Ponto onde queremos calcular a interpolação
x_ponto = 2.5

# Calcula a interpolação de Newton no ponto x_ponto
resultado_interpolado = polinomio_newton(tabela_diff, x_vals, x_ponto)

# Exibe os resultados
print(f"A tabela de diferenças divididas é:\n{tabela_diff}")
print(f"O valor interpolado no ponto x = {x_ponto} é: {resultado_interpolado:.6f}")
