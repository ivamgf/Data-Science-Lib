import math

def calcular_velocidade_torricelli(v0, a, delta_x):
    # Calcula a velocidade final usando a equação de Torricelli
    v = math.sqrt(v0**2 + 2 * a * delta_x)
    return v

# Exemplo de uso
v0 = float(input("Digite a velocidade inicial (v0): "))  # em metros por segundo
a = float(input("Digite a aceleração (a): "))  # em metros por segundo ao quadrado
delta_x = float(input("Digite o deslocamento (Δx): "))  # em metros

v = calcular_velocidade_torricelli(v0, a, delta_x)

print(f"A velocidade final da partícula é {v:.2f} m/s.")
