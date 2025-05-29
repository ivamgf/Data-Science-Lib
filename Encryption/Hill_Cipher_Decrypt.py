# Hill Cipher Decrypt

import numpy as np


# Converte letras para números
def texto_para_numeros(texto):
    texto = texto.upper().replace(" ", "")
    return [ord(c) - ord('A') for c in texto]


# Converte números para letras
def numeros_para_texto(numeros):
    return ''.join([chr(n % 26 + ord('A')) for n in numeros])


# Divide em blocos
def dividir_blocos(lista, tamanho):
    return [lista[i:i + tamanho] for i in range(0, len(lista), tamanho)]


# Inverso modular
def inverso_modular(a, m):
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    raise ValueError("Sem inverso modular")


# Inversa da matriz 2x2 mod 26
def inversa_matriz_mod26(matriz):
    det = int(round(np.linalg.det(matriz))) % 26
    if np.gcd(det, 26) != 1:
        raise ValueError(f"Determinante {det} não tem inverso módulo 26")

    det_inv = inverso_modular(det, 26)
    adj = np.array([[matriz[1][1], -matriz[0][1]],
                    [-matriz[1][0], matriz[0][0]]]) % 26
    return (det_inv * adj) % 26


# Descriptografar Hill
def descriptografar_hill(cifrado, chave):
    numeros = texto_para_numeros(cifrado)
    blocos = dividir_blocos(numeros, chave.shape[0])
    chave_inversa = inversa_matriz_mod26(chave)

    texto_claro = []
    for bloco in blocos:
        vetor = np.array(bloco)
        resultado = np.dot(chave_inversa, vetor) % 26
        texto_claro.extend(resultado)

    return numeros_para_texto(texto_claro)


# Função principal
def main():
    texto_cifrado = input("Digite o texto cifrado (ex: GXKQYTLBNSUP): ")

    # Matriz chave usada para cifragem
    chave = np.array([[3, 3],
                      [2, 5]])

    try:
        texto_original = descriptografar_hill(texto_cifrado, chave)
        print(f"Mensagem descriptografada: {texto_original}")
    except ValueError as e:
        print("Erro:", e)


main()
