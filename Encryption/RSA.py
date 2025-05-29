# RSA Algorithm Encrypt

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


# Função para a criptografia e descriptografia RSA
def rsa():
    # Pergunta se o usuário deseja ver uma lista de números primos
    show_primes = input("Você deseja ver uma lista de números primos para escolher? (sim/não): ").strip().lower()

    if show_primes == "sim":
        primos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        print("\nEscolha um número primo para p e q da lista abaixo:")
        print(primos)
    else:
        primos = []

    # Passo 1: Solicitar os números primos p e q, ou escolher da lista
    p = int(input("Digite o número primo p: ")) if not primos else int(input("Escolha um número primo para p: "))
    q = int(input("Digite o número primo q: ")) if not primos else int(input("Escolha um número primo para q: "))

    # Passo 2: Calcular n
    n = p * q

    # Passo 3: Calcular o totiente de Euler
    phi = (p - 1) * (q - 1)

    # Passo 4: Escolher e tal que 1 < e < φ(n) e mdc(e, φ(n)) = 1
    e = int(input(f"Escolha o número e tal que 1 < e < {phi} e mdc(e, φ(n)) = 1: "))
    while euclides(e, phi) != 1:
        e = int(input(f"Escolha novamente o número e tal que 1 < e < {phi} e mdc(e, φ(n)) = 1: "))

    # Passo 5: Calcular o inverso modular de e
    d = inverso_modular(e, phi)

    # Chaves públicas e privadas
    public_key = (e, n)
    private_key = (d, n)

    print(f"\nChave pública: {public_key}")
    print(f"Chave privada: {private_key}")

    # Passo 6: Criptografar a mensagem
    message = input("\nDigite a mensagem para criptografar: ")
    message_num = [ord(char) for char in message]  # Converte a mensagem para números

    # Criptografia
    cipher_text = [pow(m, e, n) for m in message_num]  # c = m^e mod n
    print("\nTexto criptografado:", cipher_text)


# Chama a função RSA para gerar chaves e criptografar a mensagem
rsa()
