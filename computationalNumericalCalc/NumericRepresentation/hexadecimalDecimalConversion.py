def hexadecimal_para_decimal(hexadecimal):
    decimal = 0
    hexadecimal = hexadecimal.upper()  # Converte para maiúsculas para facilitar a conversão

    for i, digit in enumerate(reversed(hexadecimal)):
        if '0' <= digit <= '9':
            decimal += int(digit) * (16 ** i)
        elif 'A' <= digit <= 'F':
            decimal += (ord(digit) - ord('A') + 10) * (16 ** i)
        else:
            raise ValueError("Número hexadecimal inválido.")

    return decimal


# Exemplo de uso:
numero_hexadecimal = input("Digite um número hexadecimal: ")

try:
    decimal = hexadecimal_para_decimal(numero_hexadecimal)
    print(f"O número decimal equivalente é: {decimal}")
except ValueError as e:
    print(e)
