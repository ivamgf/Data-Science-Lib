def calcular_momento_linear(massa, velocidade):
    momento_linear = massa * velocidade
    return momento_linear

# Exemplo de uso:
massa = float(input("Digite a massa do corpo (em kg): "))
velocidade = float(input("Digite a velocidade do corpo (em m/s): "))

momento_linear = calcular_momento_linear(massa, velocidade)
print(f"O momento linear é {momento_linear} kg·m/s")
