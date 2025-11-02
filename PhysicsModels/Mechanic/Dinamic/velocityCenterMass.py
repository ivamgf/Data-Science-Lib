def calcular_velocidade_centro_de_massa(particulas):
    soma_mas = 0
    soma_vx = 0
    soma_vy = 0
    soma_vz = 0

    for particula in particulas:
        massa, v_x, v_y, v_z = particula
        soma_mas += massa
        soma_vx += massa * v_x
        soma_vy += massa * v_y
        soma_vz += massa * v_z

    v_cm_x = soma_vx / soma_mas
    v_cm_y = soma_vy / soma_mas
    v_cm_z = soma_vz / soma_mas

    return v_cm_x, v_cm_y, v_cm_z

# Exemplo de uso:
n = int(input("Digite o número de partículas: "))

particulas = []
for i in range(n):
    massa = float(input(f"Digite a massa da partícula {i+1} (em kg): "))
    v_x = float(input(f"Digite a velocidade na direção x da partícula {i+1} (em m/s): "))
    v_y = float(input(f"Digite a velocidade na direção y da partícula {i+1} (em m/s): "))
    v_z = float(input(f"Digite a velocidade na direção z da partícula {i+1} (em m/s): "))
    particulas.append((massa, v_x, v_y, v_z))

velocidade_centro_de_massa = calcular_velocidade_centro_de_massa(particulas)
print(f"A velocidade do centro de massa do sistema é: (v_cm_x: {velocidade_centro_de_massa[0]}, v_cm_y: {velocidade_centro_de_massa[1]}, v_cm_z: {velocidade_centro_de_massa[2]})")
