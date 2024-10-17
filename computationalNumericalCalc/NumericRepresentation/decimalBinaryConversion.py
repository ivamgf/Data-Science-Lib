def decimal_para_binario(decimal):
    if decimal == 0:
        return "0"

    binario = ""
    while decimal > 0:
        resto = decimal % 2
        binario = str(resto) + binario  # Adiciona o resto à frente da string binária
        decimal //= 2  # Divide o decimal por 2, descartando a parte decimal

    return binario


# Exemplo de uso:
numero_decimal = int(input("Digite um número decimal: "))

binario = decimal_para_binario(numero_decimal)
print(f"O número binário equivalente é: {binario}")
