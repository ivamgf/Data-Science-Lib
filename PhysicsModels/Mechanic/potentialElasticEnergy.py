def calcular_energia_potencial_elastica(constante_mola, deformacao):
    energia_potencial_elastica = 0.5 * constante_mola * (deformacao ** 2)
    return energia_potencial_elastica

# Exemplo de uso:
constante_mola = float(input("Digite a constante elástica da mola (em N/m): "))
deformacao = float(input("Digite a deformação da mola (em metros): "))

energia_potencial_elastica = calcular_energia_potencial_elastica(constante_mola, deformacao)
print(f"A energia potencial elástica é {energia_potencial_elastica} Joules")
