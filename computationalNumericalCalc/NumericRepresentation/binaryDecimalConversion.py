def binario_para_decimal(binario):
    decimal = 0
    potencia = 0

    # Inverte a string do número binário para processar do último bit para o primeiro
    binario_invertido = binario[::-1]

    for bit in binario_invertido:
        if bit == '1':
            decimal += 2 ** potencia
        potencia += 1

    return decimal


# Exemplo de uso:
numero_binario = input("Digite um número binário: ")
# Valida se o número binário está correto
if all(bit in '01' for bit in numero_binario):
    decimal = binario_para_decimal(numero_binario)
    print(f"O número decimal equivalente é: {decimal}")
else:
    print("Por favor, digite um número binário válido.")
