# Massa de probabilidade para distribuição binomial

import math

def combinacao(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def fmp_binomial(n, p, k):
    """
    Calcula a função massa de probabilidade (FMP) binomial.
    :param n: número de tentativas
    :param p: probabilidade de sucesso
    :param k: número de sucessos desejado
    :return: probabilidade de obter k sucessos em n tentativas
    """
    q = 1 - p
    comb = combinacao(n, k)
    prob = comb * (p ** k) * (q ** (n - k))
    return prob

# Exemplo: Probabilidade de obter exatamente 3 caras (sucessos) em 5 lançamentos de uma moeda justa (p = 0.5)
n = 5
p = 5/9
k = 3

print(f"P(X = {k}) = {fmp_binomial(n, p, k):.4f}")
