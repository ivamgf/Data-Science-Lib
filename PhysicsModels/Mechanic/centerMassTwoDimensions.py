def calcular_centro_de_massa(particulas):
    soma_mas = 0
    soma_x = 0
    soma_y = 0

    for particula in particulas:
        massa, x, y = particula
        soma_mas += massa
        soma_x += massa * x
        soma_y += massa * y

    x_cm = soma_x / soma_mas
    y_cm = soma_y / soma_mas

    return x_cm, y_cm

# Exemplo de uso:
n = int(input("Digite o número de partículas: "))

particulas = []
for i in range(n):
    massa = float(input(f"Digite a massa da partícula {i+1} (em kg): "))
    x = float(input(f"Digite a coordenada x da partícula {i+1} (em metros): "))
    y = float(input(f"Digite a coordenada y da partícula {i+1} (em metros): "))
    particulas.append((massa, x, y))

centro_de_massa = calcular_centro_de_massa(particulas)
print(f"O centro de massa do sistema é: (x_cm: {centro_de_massa[0]}, y_cm: {centro_de_massa[1]})")
