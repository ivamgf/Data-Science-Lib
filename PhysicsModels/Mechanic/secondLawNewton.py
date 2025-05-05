def calcular_forca(massa, aceleracao):
    return massa * aceleracao

# Exemplo de uso:
massa = float(input("Digite a massa do objeto (em kg): "))
aceleracao = float(input("Digite a aceleração do objeto (em m/s²): "))

forca = calcular_forca(massa, aceleracao)
print(f"A força aplicada é {forca} N")
