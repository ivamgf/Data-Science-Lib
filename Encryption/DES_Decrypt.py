# DES Decrypt

# DES Decrypt com entrada da chave e da mensagem cifrada via console

from des import DesKey
import binascii

# Função para remover o padding manual (PKCS5)
def unpad(text_bytes):
    padding_len = text_bytes[-1]
    return text_bytes[:-padding_len]

# Entrada da chave e da mensagem cifrada (em hexadecimal)
chave_hex = input("Digite a chave em hexadecimal (16 caracteres): ").strip()
mensagem_cifrada_hex = input("Digite a mensagem cifrada em hexadecimal: ").strip()

# Conversão de hex para bytes
try:
    chave = bytes.fromhex(chave_hex)
    mensagem_cifrada = bytes.fromhex(mensagem_cifrada_hex)
except ValueError:
    print("Erro: valores hexadecimais inválidos.")
    exit(1)

# Verificação do tamanho da chave
if len(chave) != 8:
    print("Erro: a chave precisa ter exatamente 8 bytes (16 caracteres hex).")
    exit(1)

# Criar objeto DesKey
key = DesKey(chave)

# Descriptografar (modo ECB)
mensagem_preenchida = key.decrypt(mensagem_cifrada, padding=False)

# Remover padding
try:
    mensagem_original = unpad(mensagem_preenchida)
    print("\n Mensagem descriptografada:")
    print(mensagem_original.decode('utf-8'))
except Exception as e:
    print("Erro ao remover padding ou decodificar mensagem:", str(e))

