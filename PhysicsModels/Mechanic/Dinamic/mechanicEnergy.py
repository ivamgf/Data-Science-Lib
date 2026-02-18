def calcular_energia_cinetica(massa, velocidade):
    return 0.5 * massa * (velocidade ** 2)

def calcular_energia_potencial(massa, altura, gravidade=9.8):
    return massa * gravidade * altura

def calcular_energia_mecanica(massa, velocidade, altura):
    energia_cinetica = calcular_energia_cinetica(massa, velocidade)
    energia_potencial = calcular_energia_potencial(massa, altura)
    energia_mecanica = energia_cinetica + energia_potencial
    return energia_mecanica

# Exemplo de uso:
massa = float(input("Digite a massa do corpo (em kg): "))
velocidade = float(input("Digite a velocidade do corpo (em m/s): "))
altura = float(input("Digite a altura do corpo (em metros): "))

energia_mecanica = calcular_energia_mecanica(massa, velocidade, altura)
print(f"A energia mecânica total é {energia_mecanica} Joules")
