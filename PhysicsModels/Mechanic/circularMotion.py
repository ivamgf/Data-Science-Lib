import math


def movimento_circular(r, T, t, theta_0=0):
    # Calcula a velocidade angular (omega)
    omega = 2 * math.pi / T

    # Calcula a velocidade tangencial
    v = r * omega

    # Calcula a aceleração centrípeta
    a_c = omega ** 2 * r

    # Calcula a posição angular no tempo t
    theta = theta_0 + omega * t

    return omega, v, a_c, theta


# Exemplo de uso
r = float(input("Digite o raio da trajetória (r) em metros: "))  # em metros
T = float(input("Digite o período (T) em segundos: "))  # em segundos
t = float(input("Digite o tempo (t) em segundos: "))  # em segundos
theta_0 = float(input("Digite o ângulo inicial (θ_0) em radianos: "))  # em radianos (opcional)

# Cálculo
omega, v, a_c, theta = movimento_circular(r, T, t, theta_0)

# Exibe os resultados
print(f"A velocidade angular é {omega:.2f} rad/s.")
print(f"A velocidade tangencial é {v:.2f} m/s.")
print(f"A aceleração centrípeta é {a_c:.2f} m/s².")
print(f"A posição angular após {t} segundos é {theta:.2f} radianos.")
