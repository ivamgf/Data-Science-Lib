# AES Decrypt with IV (Initialization Vector)

# Imports
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import binascii

# Entrada do usuário
mensagem_cifrada_hex = input("Digite a mensagem cifrada (hex): ")
chave_hex = input("Digite a chave AES (hex, 16 bytes = 32 caracteres): ")
iv_hex = input("Digite o IV (hex, 16 bytes = 32 caracteres): ")

# Conversão de hex para bytes
try:
    mensagem_cifrada = bytes.fromhex(mensagem_cifrada_hex)
    chave = bytes.fromhex(chave_hex)
    iv = bytes.fromhex(iv_hex)
except ValueError:
    print("Erro: Entrada em hexadecimal inválida.")
    exit(1)

# Verificações básicas
if len(chave) != 16 or len(iv) != 16:
    print("Erro: A chave e o IV devem ter exatamente 16 bytes (32 caracteres hex).")
    exit(1)

# Criação do objeto Cipher
cipher = Cipher(algorithms.AES(chave), modes.CBC(iv), backend=default_backend())
decryptor = cipher.decryptor()

# Descriptografar
mensagem_preenchida = decryptor.update(mensagem_cifrada) + decryptor.finalize()

# Remover padding
unpadder = padding.PKCS7(128).unpadder()
mensagem_original = unpadder.update(mensagem_preenchida) + unpadder.finalize()

# Exibir resultado
print("\nMensagem descriptografada:")
print(mensagem_original.decode('utf-8'))
