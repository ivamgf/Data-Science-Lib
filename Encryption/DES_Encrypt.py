# DES Encrypt

from Crypto.Cipher import DES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

# Mensagem original
mensagem = "A senha é ilha1389"
mensagem_bytes = mensagem.encode('utf-8')

# Geração de chave de 8 bytes
chave = get_random_bytes(8)

# Criar objeto de cifra (modo ECB)
cipher = DES.new(chave, DES.MODE_ECB)

# Padding da mensagem para múltiplos de 8
mensagem_preenchida = pad(mensagem_bytes, DES.block_size)

# Criptografar
mensagem_cifrada = cipher.encrypt(mensagem_preenchida)

# Salvar a chave e mensagem criptografada em arquivos binários
with open("chave_des.bin", "wb") as chave_arquivo:
    chave_arquivo.write(chave)

with open("mensagem_cifrada_des.bin", "wb") as msg_arquivo:
    msg_arquivo.write(mensagem_cifrada)

print("Mensagem criptografada e salva com sucesso!")
print("Chave (hex):", chave.hex())
print("Mensagem cifrada (hex):", mensagem_cifrada.hex())
