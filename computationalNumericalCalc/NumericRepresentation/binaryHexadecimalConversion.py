def binario_para_hexadecimal(binario):
    # Certifica-se de que o número binário possui um comprimento múltiplo de 4
    while len(binario) % 4 != 0:
        binario = '0' + binario  # Adiciona zeros à esquerda

    hexadecimal = ""

    # Mapeamento de grupos binários para hexadecimais
    binario_para_hex = {
        '0000': '0',
        '0001': '1',
        '0010': '2',
        '0011': '3',
        '0100': '4',
        '0101': '5',
        '0110': '6',
        '0111': '7',
        '1000': '8',
        '1001': '9',
        '1010': 'A',
        '1011': 'B',
        '1100': 'C',
        '1101': 'D',
        '1110': 'E',
        '1111': 'F',
    }

    # Divide o número binário em grupos de 4 e converte
    for i in range(0, len(binario), 4):
        grupo = binario[i:i + 4]
        hexadecimal += binario_para_hex[grupo]

    return hexadecimal


# Exemplo de uso:
numero_binario = input("Digite um número binário: ")
# Valida se o número binário está correto
if all(bit in '01' for bit in numero_binario):
    hexadecimal = binario_para_hexadecimal(numero_binario)
    print(f"O número hexadecimal equivalente é: {hexadecimal}")
else:
    print("Por favor, digite um número binário válido.")
