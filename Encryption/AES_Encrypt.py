# AES Encrypt

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

# Mensagem original
mensagem = "A senha é ilha1389"
mensagem_bytes = mensagem.encode('utf-8')

# Geração da chave AES de 16 bytes (128 bits)
chave = get_random_bytes(16)

# Geração do IV (Initialization Vector) de 16 bytes
iv = get_random_bytes(16)

# Criação do objeto AES com modo CBC
cipher = AES.new(chave, AES.MODE_CBC, iv)

# Padding da mensagem para múltiplos de 16 bytes
mensagem_preenchida = pad(mensagem_bytes, AES.block_size)

# Criptografar
mensagem_cifrada = cipher.encrypt(mensagem_preenchida)

# Salvar a chave, IV e mensagem cifrada em arquivos binários
with open("chave_aes.bin", "wb") as chave_file:
    chave_file.write(chave)

with open("iv_aes.bin", "wb") as iv_file:
    iv_file.write(iv)

with open("mensagem_cifrada_aes.bin", "wb") as msg_file:
    msg_file.write(mensagem_cifrada)

print("Mensagem criptografada com sucesso!")
print("Chave (hex):", chave.hex())
print("IV (hex):", iv.hex())
print("Mensagem cifrada (hex):", mensagem_cifrada.hex())
