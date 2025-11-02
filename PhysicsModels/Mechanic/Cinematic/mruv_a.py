def calcular_aceleracao(v0, v, t):
    # Calcula a aceleração no MRUV
    if t != 0:
        a = (v - v0) / t
        return a
    else:
        return "O tempo não pode ser zero."

# Exemplo de uso
v0 = float(input("Digite a velocidade inicial (v0): "))  # em metros por segundo
v = float(input("Digite a velocidade final (v): "))  # em metros por segundo
t = float(input("Digite o tempo (t): "))  # em segundos

a = calcular_aceleracao(v0, v, t)

if isinstance(a, str):
    print(a)  # Caso o tempo seja zero, exibe a mensagem de erro
else:
    print(f"A aceleração da partícula é {a:.2f} m/s².")
