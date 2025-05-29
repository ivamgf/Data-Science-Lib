# RSA Algorithm Decrypt

# Função para calcular o mdc usando o Algoritmo de Euclides
def euclides(a, b):
    while b:
        a, b = b, a % b
    return a

# Função para calcular o inverso modular usando o Algoritmo de Euclides Estendido
def inverso_modular(e, phi):
    t, new_t = 0, 1
    r, new_r = phi, e
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError(f'{e} não tem inverso modular em relação a {phi}')
    if t < 0:
        t = t + phi
    return t

# Função para descriptografar RSA
def descriptografar(cipher_text, d, n):
    decrypted_message = [chr(pow(c, d, n)) for c in cipher_text]  # m = c^d mod n
    return ''.join(decrypted_message)

# Função para a descriptografia RSA
def rsa_descriptografar():
    # Passo 1: Solicitar a chave privada
    d = int(input("Digite o valor de d (parte da chave privada): "))
    n = int(input("Digite o valor de n (parte da chave privada): "))

    # Passo 2: Solicitar o texto cifrado
    cipher_text = input("Digite o texto cifrado (como uma lista de números, por exemplo [123, 456, 789]): ")
    cipher_text = eval(cipher_text)  # Converte a entrada em uma lista de inteiros

    # Passo 3: Descriptografar a mensagem
    decrypted_message = descriptografar(cipher_text, d, n)

    # Exibir a mensagem descriptografada
    print("\nMensagem descriptografada:", decrypted_message)

# Chama a função para descriptografar a mensagem
rsa_descriptografar()
