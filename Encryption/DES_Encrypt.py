# DES Encrypt

# Imports
# DES Encrypt com entrada do usuário

from des import DesKey
import os

# Entrada da mensagem pelo usuário
mensagem = input("Digite a mensagem a ser criptografada: ")
mensagem_bytes = mensagem.encode('utf-8')

# Geração de chave de 8 bytes (64 bits)
chave = os.urandom(8)

# Criar objeto de chave DES
key = DesKey(chave)

# Padding manual para múltiplos de 8 bytes (PKCS5)
def pad(text_bytes):
    padding_len = 8 - len(text_bytes) % 8
    return text_bytes + bytes([padding_len] * padding_len)

mensagem_preenchida = pad(mensagem_bytes)

# Criptografar (modo ECB por padrão)
mensagem_cifrada = key.encrypt(mensagem_preenchida, padding=False)

# Salvar a chave e a mensagem cifrada em arquivos binários
with open("chave_des.bin", "wb") as chave_arquivo:
    chave_arquivo.write(chave)

with open("mensagem_cifrada_des.bin", "wb") as msg_arquivo:
    msg_arquivo.write(mensagem_cifrada)

# Saída
print("\n Mensagem criptografada e salva com sucesso!")
print("Chave (hex):", chave.hex())
print("Mensagem cifrada (hex):", mensagem_cifrada.hex())
