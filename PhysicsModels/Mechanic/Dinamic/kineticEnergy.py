def calcular_energia_cinetica(massa, velocidade):
    energia_cinetica = 0.5 * massa * (velocidade ** 2)
    return energia_cinetica

# Exemplo de uso:
massa = float(input("Digite a massa do corpo (em kg): "))
velocidade = float(input("Digite a velocidade do corpo (em m/s): "))

energia_cinetica = calcular_energia_cinetica(massa, velocidade)
print(f"A energia cinética é {energia_cinetica} Joules")
