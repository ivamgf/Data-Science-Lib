# Probabilidade com combinatória para 3 ou mais eventos simultâneos

import math

def combinacao(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def probabilidade_eventos_simultaneos_combinatoria(total1, favoravel1, k1, total2, favoravel2, k2):
    """
    Calcula a probabilidade de dois eventos simultâneos (independentes) usando combinações.
    :param total1: total de elementos no primeiro evento
    :param favoravel1: casos favoráveis no primeiro evento
    :param k1: número de elementos escolhidos no primeiro evento
    :param total2: total de elementos no segundo evento
    :param favoravel2: casos favoráveis no segundo evento
    :param k2: número de elementos escolhidos no segundo evento
    :return: probabilidade conjunta
    """
    casos_possiveis1 = combinacao(total1, k1)
    casos_favoraveis1 = combinacao(favoravel1, k1)

    casos_possiveis2 = combinacao(total2, k2)
    casos_favoraveis2 = combinacao(favoravel2, k2)

    prob1 = casos_favoraveis1 / casos_possiveis1
    prob2 = casos_favoraveis2 / casos_possiveis2

    return prob1 * prob2

# Exemplo: probabilidade de tirar 1 carta vermelha de 52 (26 favoráveis) e 1 Ás (4 de 52 cartas)
print("Probabilidade de eventos simultâneos: {:.4f}".format(
    probabilidade_eventos_simultaneos_combinatoria(52, 26, 1, 52, 4, 1)
))
