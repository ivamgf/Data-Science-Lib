# XOR Decrypt

# XOR - Descriptografia com chave salva
import base64

# Solicitar ao usuário a mensagem criptografada (em Base64)
cipher_base64 = input("Digite a mensagem criptografada (Base64): ")

try:
    # Decodificar a mensagem Base64 para bytes
    cipher_bytes = base64.b64decode(cipher_base64)
    cipher = list(cipher_bytes)

    # Ler o arquivo com a chave salva
    with open("xor_key.bin", "rb") as f_key:
        key = list(f_key.read())

    # Verificar se os tamanhos coincidem
    if len(cipher) != len(key):
        print("\nErro: O tamanho da chave não corresponde ao tamanho da mensagem cifrada.")
    else:
        # Mostrar a mensagem cifrada em binário
        binary_cipher = ' '.join(format(byte, '08b') for byte in cipher)

        # Realizar descriptografia com XOR
        decrypted = ''.join([chr(c ^ k) for c, k in zip(cipher, key)])

        # Exibir os resultados
        print("\nMensagem criptografada (binário):")
        print(binary_cipher)

        print("\nMensagem descriptografada:")
        print(decrypted)

except Exception as e:
    print("\nErro ao processar a mensagem criptografada:", e)
