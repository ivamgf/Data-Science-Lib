def lancamento_horizontal(v0, y0, t, g=9.8):
    # Componente horizontal da posição e velocidade (constante)
    x = v0 * t
    v_x = v0

    # Componente vertical da posição e velocidade
    y = y0 - 0.5 * g * t ** 2
    v_y = g * t

    return x, y, v_x, v_y


# Exemplo de uso
v0 = float(input("Digite a velocidade inicial horizontal (v0) em m/s: "))  # em metros por segundo
y0 = float(input("Digite a altura inicial (y0) em metros: "))  # em metros
t = float(input("Digite o tempo (t) em segundos: "))  # em segundos

# Cálculo
x, y, v_x, v_y = lancamento_horizontal(v0, y0, t)

# Exibe os resultados
print(f"A posição horizontal após {t} segundos é {x:.2f} metros.")
print(f"A posição vertical após {t} segundos é {y:.2f} metros.")
print(f"A velocidade horizontal após {t} segundos é {v_x:.2f} m/s.")
print(f"A velocidade vertical após {t} segundos é {v_y:.2f} m/s.")
