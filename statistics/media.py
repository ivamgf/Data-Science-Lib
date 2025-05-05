# Media

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
print(f"A média aritmética é: {media}")
