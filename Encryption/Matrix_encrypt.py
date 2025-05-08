# Matrix Encrypt

import numpy as np

def mensagem_para_numeros(mensagem):
    return [ord(c) for c in mensagem]

def quebrar_em_blocos(lista, tamanho):
    while len(lista) % tamanho != 0:
        lista.append(0)
    return [lista[i:i+tamanho] for i in range(0, len(lista), tamanho)]

def criptografar_matriz(mensagem, chave):
    numeros = mensagem_para_numeros(mensagem)
    blocos = quebrar_em_blocos(numeros, chave.shape[0])
    criptografado = []

    for bloco in blocos:
        bloco_np = np.array(bloco)
        resultado = np.dot(chave, bloco_np) % 256
        criptografado.append(resultado)

    return criptografado

def imprimir_blocos(blocos):
    print("\nBlocos criptografados:")
    for b in blocos:
        print(b)

mensagem = input("Digite a mensagem para criptografar: ")
chave = np.array([[1, 1], [1, 0]])  # matriz com determinante 255

resultado = criptografar_matriz(mensagem, chave)
imprimir_blocos(resultado)

