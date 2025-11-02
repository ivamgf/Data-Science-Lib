def calcular_momento_linear_total(massas, velocidades):
    momento_total = 0
    for i in range(len(massas)):
        momento_total += massas[i] * velocidades[i]
    return momento_total

# Exemplo de uso:
n = int(input("Digite o número de objetos: "))

massas = []
velocidades = []

for i in range(n):
    massa = float(input(f"Digite a massa do objeto {i+1} (em kg): "))
    velocidade = float(input(f"Digite a velocidade do objeto {i+1} (em m/s): "))
    massas.append(massa)
    velocidades.append(velocidade)

momento_total = calcular_momento_linear_total(massas, velocidades)
print(f"O momento linear total é {momento_total} kg·m/s")
