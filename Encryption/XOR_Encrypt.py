# XOR Encrypt

# XOR - Criptografia com chave pseudoaleatória baseada no tempo
import time
import random
import base64

# Mensagem original
message = input("Digite a mensagem a ser criptografada: ")

# Gerar semente baseada no tempo atual
random.seed(time.time())

# Gerar chave pseudoaleatória (mesmo tamanho da mensagem)
key = [random.randint(0, 255) for _ in message]

# Criptografar a mensagem com XOR
cipher = [ord(c) ^ k for c, k in zip(message, key)]
cipher_bytes = bytes(cipher)

# Salvar o texto cifrado e a chave em arquivos binários
with open("xor_cipher.bin", "wb") as f_cipher:
    f_cipher.write(cipher_bytes)

with open("xor_key.bin", "wb") as f_key:
    f_key.write(bytes(key))

# Mostrar resultado da cifra em binário (bit a bit)
binary_cipher = ' '.join(format(byte, '08b') for byte in cipher)

# Mostrar resultado em alfanumérico (base64 para legibilidade)
cipher_base64 = base64.b64encode(cipher_bytes).decode('utf-8')

# Exibir resultados
print("Mensagem original:", message)
print("Mensagem criptografada (binário):")
print(binary_cipher)
print("\nMensagem criptografada (alfanumérica - Base64):")
print(cipher_base64)
