import math


def lancamento_vertical(v0, y0, t, g=9.8):
    # Aceleração no lançamento vertical
    a = -g

    # Calcula a posição no instante t
    y = y0 + v0 * t + 0.5 * a * t ** 2

    # Calcula a velocidade no instante t
    v = v0 + a * t

    return y, v


# Exemplo de uso
v0 = float(input("Digite a velocidade inicial (v0) em m/s: "))  # em metros por segundo
y0 = float(input("Digite a posição inicial (y0) em metros: "))  # em metros
t = float(input("Digite o tempo (t) em segundos: "))  # em segundos

# Cálculo
y, v = lancamento_vertical(v0, y0, t)

# Exibe os resultados
print(f"A posição da partícula após {t} segundos é {y:.2f} metros.")
print(f"A velocidade da partícula após {t} segundos é {v:.2f} m/s.")
