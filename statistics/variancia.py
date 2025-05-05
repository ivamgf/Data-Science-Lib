## Média, Desvios, Desvios Quadráticos e Variância

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

# Cálculo da variância (dividido por n-1)
if n > 1:
    variancia = (sum(desvios_quadraticos) / (n - 1)) ** 2
    print(f"\nVariância: {variancia}")
else:
    print("\nNão é possível calcular a variância com menos de 2 elementos.")
