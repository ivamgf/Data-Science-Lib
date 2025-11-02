def calcular_velocidade_mruv(v0, a, t):
    # Calcula a velocidade no MRUV
    v = v0 + a * t
    return v

# Exemplo de uso
v0 = float(input("Digite a velocidade inicial (v0): "))  # em metros por segundo
a = float(input("Digite a aceleração (a): "))  # em metros por segundo ao quadrado
t = float(input("Digite o tempo (t): "))  # em segundos

v = calcular_velocidade_mruv(v0, a, t)

print(f"A velocidade da partícula após {t} segundos é {v:.2f} m/s.")
