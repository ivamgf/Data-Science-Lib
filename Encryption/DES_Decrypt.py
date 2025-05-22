# DES Decrypt

from Crypto.Cipher import DES
from Crypto.Util.Padding import unpad

# Ler chave do arquivo
with open("chave_des.bin", "rb") as chave_arquivo:
    chave = chave_arquivo.read()

# Ler mensagem cifrada do arquivo
with open("mensagem_cifrada_des.bin", "rb") as msg_arquivo:
    mensagem_cifrada = msg_arquivo.read()

# Criar objeto de cifra com a chave lida
cipher = DES.new(chave, DES.MODE_ECB)

# Descriptografar
mensagem_decifrada = unpad(cipher.decrypt(mensagem_cifrada), DES.block_size)

print("Mensagem descriptografada:", mensagem_decifrada.decode('utf-8'))
