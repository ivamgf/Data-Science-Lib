# Probabilidade 2 com combinatória e evento único

import math

def combinacao(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def probabilidade_evento_unico(n_total, n_evento):
    """
    Calcula a probabilidade de um evento único usando a fórmula da combinação.
    :param n_total: número total de elementos
    :param n_evento: número de elementos favoráveis ao evento
    :return: probabilidade do evento
    """
    casos_possiveis = combinacao(n_total, 1)
    casos_favoraveis = combinacao(n_evento, 1)
    return casos_favoraveis / casos_possiveis

# Exemplo: probabilidade de tirar uma carta vermelha (26) de um baralho de 52 cartas
print(f"Probabilidade de evento único: {probabilidade_evento_unico(52, 26):.2f}")
