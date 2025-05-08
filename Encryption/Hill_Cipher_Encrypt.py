# Hill Cipher Encrypt

import numpy as np


# Converte letras para números (A=0, B=1, ..., Z=25)
def texto_para_numeros(texto):
    texto = texto.upper().replace(" ", "")
    return [ord(c) - ord('A') for c in texto]


# Converte números para letras
def numeros_para_texto(numeros):
    return ''.join([chr(n % 26 + ord('A')) for n in numeros])


# Divide em blocos do tamanho da chave
def dividir_blocos(lista, tamanho):
    while len(lista) % tamanho != 0:
        lista.append(0)  # Padding com 'A' (valor 0)
    return [lista[i:i + tamanho] for i in range(0, len(lista), tamanho)]


# Criptografa usando Hill Cipher
def criptografar_hill(texto, chave):
    numeros = texto_para_numeros(texto)
    blocos = dividir_blocos(numeros, chave.shape[0])
    criptografado = []

    for bloco in blocos:
        vetor = np.array(bloco)
        resultado = np.dot(chave, vetor) % 26
        criptografado.extend(resultado)

    return numeros_para_texto(criptografado)


# Função principal
def main():
    texto = input("Digite o texto para criptografar (somente letras): ")

    # Exemplo de chave 2x2: deve ser invertível módulo 26
    chave = np.array([[3, 3],
                      [2, 5]])

    # Verifica se a chave tem determinante inversível mod 26
    det = int(np.round(np.linalg.det(chave))) % 26
    if np.gcd(det, 26) != 1:
        print(f"Chave inválida. Determinante {det} não tem inverso mod 26.")
        return

    criptografado = criptografar_hill(texto, chave)
    print(f"Texto criptografado: {criptografado}")


main()
