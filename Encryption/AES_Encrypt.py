# AES Encrypt

# Imports
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

# Entrada do usuário
mensagem = input("Digite a mensagem a ser criptografada: ")
mensagem_bytes = mensagem.encode('utf-8')

# Geração da chave AES de 16 bytes (128 bits)
chave = os.urandom(16)

# Geração do IV (Initialization Vector) de 16 bytes
iv = os.urandom(16)

# Padding (PKCS7 para blocos de 128 bits)
padder = padding.PKCS7(128).padder()
mensagem_preenchida = padder.update(mensagem_bytes) + padder.finalize()

# Criação do objeto Cipher AES com modo CBC
cipher = Cipher(algorithms.AES(chave), modes.CBC(iv), backend=default_backend())
encryptor = cipher.encryptor()

# Criptografar
mensagem_cifrada = encryptor.update(mensagem_preenchida) + encryptor.finalize()

# Salvar chave, IV e mensagem cifrada em arquivos binários
with open("chave_aes.bin", "wb") as chave_file:
    chave_file.write(chave)

with open("iv_aes.bin", "wb") as iv_file:
    iv_file.write(iv)

with open("mensagem_cifrada_aes.bin", "wb") as msg_file:
    msg_file.write(mensagem_cifrada)

# Exibição
print("\nMensagem criptografada com sucesso!")
print("Chave (hex):", chave.hex())
print("IV (hex):", iv.hex())
print("Mensagem cifrada (hex):", mensagem_cifrada.hex())
