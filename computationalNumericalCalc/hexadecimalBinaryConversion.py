def hexadecimal_para_binario(hexadecimal):
    # Mapeamento de dígitos hexadecimais para binários
    hex_para_bin = {
        '0': '0000',
        '1': '0001',
        '2': '0010',
        '3': '0011',
        '4': '0100',
        '5': '0101',
        '6': '0110',
        '7': '0111',
        '8': '1000',
        '9': '1001',
        'A': '1010',
        'B': '1011',
        'C': '1100',
        'D': '1101',
        'E': '1110',
        'F': '1111',
    }

    binario = ""
    hexadecimal = hexadecimal.upper()  # Converte para maiúsculas para facilitar a conversão

    # Converte cada dígito hexadecimal para binário
    for digit in hexadecimal:
        if digit in hex_para_bin:
            binario += hex_para_bin[digit]
        else:
            raise ValueError("Número hexadecimal inválido.")

    return binario


# Exemplo de uso:
numero_hexadecimal = input("Digite um número hexadecimal: ")

try:
    binario = hexadecimal_para_binario(numero_hexadecimal)
    print(f"O número binário equivalente é: {binario}")
except ValueError as e:
    print(e)
