def mruv(x0, v0, a, t):
    # Calcula o deslocamento no MRUV
    x = x0 + v0 * t + 0.5 * a * t**2
    return x

# Exemplo de uso
x0 = float(input("Digite a posição inicial (x0): "))  # em metros
v0 = float(input("Digite a velocidade inicial (v0): "))  # em metros por segundo
a = float(input("Digite a aceleração (a): "))  # em metros por segundo ao quadrado
t = float(input("Digite o tempo (t): "))  # em segundos

x = mruv(x0, v0, a, t)

print(f"O deslocamento da partícula após {t} segundos é {x:.2f} metros.")
