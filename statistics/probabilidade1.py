# Probabilidade

import math

def probabilidade_evento_unico(n_total, n_evento):
    """
    Calcula a probabilidade de um único evento usando combinatória.
    :param n_total: número total de elementos (ex: total de cartas)
    :param n_evento: número de elementos favoráveis ao evento (ex: cartas vermelhas)
    :return: probabilidade do evento ocorrer
    """
    casos_possiveis = math.comb(n_total, 1)  # Combinação de 1 elemento entre os n_total
    casos_favoraveis = math.comb(n_evento, 1)  # Combinação de 1 elemento entre os favoráveis

    probabilidade = casos_favoraveis / casos_possiveis
    return probabilidade

# Exemplo: probabilidade de tirar uma carta vermelha de um baralho de 52 cartas (26 cartas vermelhas)
print(f"Probabilidade de evento único: {probabilidade_evento_unico(52, 26):.2f}")
