# Intervalo de Confiança com t de Student

import math
from scipy.stats import t

# Solicita os dados ao usuário
media = float(input("Digite a média da amostra: "))
variancia = float(input("Digite a variância da amostra: "))
n = int(input("Digite o tamanho da amostra: "))

# Nível de confiança (fixo em 95%)
nivel_confianca = 0.95
graus_liberdade = n - 1

# Calcula o valor crítico de t
t_critico = t.ppf((1 + nivel_confianca) / 2, graus_liberdade)

# Calcula o desvio padrão e o erro padrão da média
desvio_padrao = math.sqrt(variancia)
erro_padrao = desvio_padrao / math.sqrt(n)

# Calcula o intervalo de confiança
limite_inferior = media - t_critico * erro_padrao
limite_superior = media + t_critico * erro_padrao

# Exibe o resultado
print(f"\nIntervalo de confiança de 95% usando a distribuição t de Student:")
print(f"({limite_inferior:.4f}, {limite_superior:.4f})")
