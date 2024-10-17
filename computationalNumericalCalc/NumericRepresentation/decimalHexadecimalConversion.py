def decimal_para_hexadecimal(decimal):
    if decimal == 0:
        return "0"

    hexadecimal = ""
    while decimal > 0:
        resto = decimal % 16
        if resto < 10:
            hexadecimal = str(resto) + hexadecimal  # Números de 0 a 9
        else:
            hexadecimal = chr(ord('A') + (resto - 10)) + hexadecimal  # Números de 10 a 15 (A-F)
        decimal //= 16  # Divide o decimal por 16

    return hexadecimal

# Exemplo de uso:
numero_decimal = int(input("Digite um número decimal: "))

hexadecimal = decimal_para_hexadecimal(numero_decimal)
print(f"O número hexadecimal equivalente é: {hexadecimal}")
