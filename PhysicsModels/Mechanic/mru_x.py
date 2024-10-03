def mru(x0, v, t):
    # Calcula o deslocamento
    x = x0 + v * t
    return x, v

# Exemplo de uso
x0 = float(input("Digite a posição inicial (x0): "))  # em metros
v = float(input("Digite a velocidade (v): "))         # em metros por segundo
t = float(input("Digite o tempo (t): "))              # em segundos

x, v_calculada = mru(x0, v, t)

print(f"O deslocamento da partícula após {t} segundos é {x:.2f} metros.")
print(f"A velocidade da partícula é {v_calculada:.2f} m/s.")
