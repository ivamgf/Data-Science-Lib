import math


def lancamento_obliquo(v0, theta, y0, t, g=9.8):
    # Converte o ângulo para radianos
    theta_rad = math.radians(theta)

    # Componente horizontal da velocidade
    v_x = v0 * math.cos(theta_rad)

    # Componente vertical da velocidade
    v_y = v0 * math.sin(theta_rad) - g * t

    # Posição horizontal no tempo t
    x = v_x * t

    # Posição vertical no tempo t
    y = y0 + v0 * math.sin(theta_rad) * t - 0.5 * g * t ** 2

    return x, y, v_x, v_y


# Exemplo de uso
v0 = float(input("Digite a velocidade inicial (v0) em m/s: "))  # em metros por segundo
theta = float(input("Digite o ângulo de lançamento (θ) em graus: "))  # em graus
y0 = float(input("Digite a posição vertical inicial (y0) em metros: "))  # em metros
t = float(input("Digite o tempo (t) em segundos: "))  # em segundos

# Cálculo
x, y, v_x, v_y = lancamento_obliquo(v0, theta, y0, t)

# Exibe os resultados
print(f"A posição horizontal após {t} segundos é {x:.2f} metros.")
print(f"A posição vertical após {t} segundos é {y:.2f} metros.")
print(f"A velocidade horizontal após {t} segundos é {v_x:.2f} m/s.")
print(f"A velocidade vertical após {t} segundos é {v_y:.2f} m/s.")
