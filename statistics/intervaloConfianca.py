# Intervalo de Confiança

import math

# Solicita os dados ao usuário
media = float(input("Digite a média da amostra: "))
variancia = float(input("Digite a variância da amostra: "))
z = float(input("Digite o valor de z (por exemplo, 1.96 para 95%): "))
n = int(input("Digite o tamanho da amostra: "))

# Calcula o desvio padrão e o erro padrão da média
desvio_padrao = math.sqrt(variancia)
erro_padrao = desvio_padrao / math.sqrt(n)

# Calcula o intervalo de confiança
limite_inferior = media - z * erro_padrao
limite_superior = media + z * erro_padrao

# Exibe o resultado
print(f"\nIntervalo de confiança de 95%:")
print(f"({limite_inferior:.4f}, {limite_superior:.4f})")
