def calcular_energia_potencial_gravitacional(massa, altura, gravidade=9.8):
    energia_potencial = massa * gravidade * altura
    return energia_potencial

# Exemplo de uso:
massa = float(input("Digite a massa do corpo (em kg): "))
altura = float(input("Digite a altura do corpo (em metros): "))

energia_potencial = calcular_energia_potencial_gravitacional(massa, altura)
print(f"A energia potencial gravitacional é {energia_potencial} Joules")
