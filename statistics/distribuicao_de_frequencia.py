## Média, Desvios, Desvios Quadráticos, Variância, Desvio Padrão e Gráfico de Frequência

import math  # necessário para usar sqrt
import matplotlib.pyplot as plt
from collections import Counter

def media_aritmetica(X):
    n = len(X)
    if n == 0:
        return None
    soma = sum(X)
    media = soma / n
    return media

# Entrada do usuário
n = int(input("Digite a quantidade de elementos (n): "))
X = []

for i in range(n):
    valor = float(input(f"Digite o valor {i+1}: "))
    X.append(valor)

# Cálculo e exibição da média
media = media_aritmetica(X)
print(f"\nA média aritmética é: {media}")

# Cálculo dos desvios e desvios quadráticos
desvios = []
desvios_quadraticos = []

print("\nResultados:")
for i in range(n):
    desvio = X[i] - media
    desvio_quadratico = desvio ** 2
    desvios.append(desvio)
    desvios_quadraticos.append(desvio_quadratico)
    print(f"Valor: {X[i]} | Desvios: {desvio} | Desvios Quadráticos: {desvio_quadratico}")

# Exibição dos arrays de desvios e desvios quadráticos
print("\nArray de Desvios:", desvios)
print("Array de Desvios Quadráticos:", desvios_quadraticos)

# Cálculo da variância e do desvio padrão
if n > 1:
    variancia = sum(desvios_quadraticos) / (n - 1)  # Corrigido aqui
    desvio_padrao = math.sqrt(variancia)
    print(f"\nVariância: {variancia}")
    print(f"Desvio Padrão: {desvio_padrao}")
else:
    print("\nNão é possível calcular a variância e o desvio padrão com menos de 2 elementos.")

# Geração do gráfico de distribuição de frequência
frequencias = Counter(X)
valores = list(frequencias.keys())
contagens = list(frequencias.values())

plt.figure(figsize=(8, 5))
plt.bar(valores, contagens, width=0.5, color='skyblue', edgecolor='black')
plt.title("Distribuição de Frequência")
plt.xlabel("Valores")
plt.ylabel("Frequência")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
