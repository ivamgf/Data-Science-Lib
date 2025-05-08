# Matrix Decrypt

import numpy as np
from math import gcd

def inverso_modular(a, m):
    a %= m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    raise ValueError(f"{a} não tem inverso módulo {m}")

def inversa_matriz_2x2_mod256(matriz):
    a, b = matriz[0]
    c, d = matriz[1]
    det = (a * d - b * c) % 256

    if gcd(det, 256) != 1:
        raise ValueError(f"O determinante {det} não tem inverso módulo 256.")
    det_inv = inverso_modular(det, 256)

    adjunta = np.array([[d, -b], [-c, a]]) % 256
    return (det_inv * adjunta) % 256

def descriptografar_matriz(blocos, chave):
    chave_inv = inversa_matriz_2x2_mod256(chave)
    mensagem_numeros = []

    for bloco in blocos:
        vetor = np.array(bloco)
        resultado = np.dot(chave_inv, vetor) % 256
        mensagem_numeros.extend(resultado)

    mensagem = ''.join(chr(int(x)) for x in mensagem_numeros if x > 0)
    return mensagem

def ler_blocos():
    print("Digite os blocos criptografados no formato: [num num], e 'fim' para encerrar:")
    blocos = []
    while True:
        entrada = input()
        if entrada.lower() == 'fim':
            break
        entrada = entrada.replace('[', '').replace(']', '').strip()
        blocos.append(list(map(int, entrada.split())))
    return blocos

def descriptografar():
    chave = np.array([[1, 1], [1, 0]])
    blocos = ler_blocos()
    mensagem = descriptografar_matriz(blocos, chave)
    print("\nMensagem descriptografada:")
    print(mensagem)

descriptografar()

