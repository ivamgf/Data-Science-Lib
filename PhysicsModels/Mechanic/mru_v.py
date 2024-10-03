def calcular_velocidade(x0, x, t):
    # Calcula a velocidade
    if t != 0:
        v = (x - x0) / t
        return v
    else:
        return "O tempo não pode ser zero."

# Exemplo de uso
x0 = float(input("Digite a posição inicial (x0): "))  # em metros
x = float(input("Digite o deslocamento final (x): ")) # em metros
t = float(input("Digite o tempo (t): "))              # em segundos

v = calcular_velocidade(x0, x, t)

if isinstance(v, str):
    print(v)  # Caso o tempo seja zero, exibe a mensagem de erro
else:
    print(f"A velocidade da partícula é {v:.2f} m/s.")
