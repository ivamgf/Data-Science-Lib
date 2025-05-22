# AES Decrypt

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# Ler chave
with open("chave_aes.bin", "rb") as chave_file:
    chave = chave_file.read()

# Ler IV
with open("iv_aes.bin", "rb") as iv_file:
    iv = iv_file.read()

# Ler mensagem cifrada
with open("mensagem_cifrada_aes.bin", "rb") as msg_file:
    mensagem_cifrada = msg_file.read()

# Criar objeto AES com a chave e o IV lidos
cipher = AES.new(chave, AES.MODE_CBC, iv)

# Descriptografar e remover padding
mensagem_decifrada = unpad(cipher.decrypt(mensagem_cifrada), AES.block_size)

print("Mensagem descriptografada:", mensagem_decifrada.decode('utf-8'))
