# FMP de Poisson

import math

def fmp_poisson(lambd, k):
    """
    Calcula a função massa de probabilidade da distribuição de Poisson.
    :param lambd: taxa média de ocorrência (λ)
    :param k: número de ocorrências desejado
    :return: probabilidade de ocorrer exatamente k eventos
    """
    return (math.exp(-lambd) * lambd**k) / math.factorial(k)

# Exemplo: Probabilidade de ocorrer exatamente 3 eventos quando λ = 2.5
lambd = 4
k = 8

print(f"P(X = {k}) = {fmp_poisson(lambd, k):.4f}")
