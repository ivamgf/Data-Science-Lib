import math


def bhaskara(a, b, c):
    # Calcula o discriminante (delta)
    delta = b ** 2 - 4 * a * c

    # Verifica se há soluções reais
    if delta < 0:
        return "A equação não possui raízes reais."

    # Calcula as raízes usando a fórmula de Bhaskara
    x1 = (-b + math.sqrt(delta)) / (2 * a)
    x2 = (-b - math.sqrt(delta)) / (2 * a)

    # Verifica se as raízes são iguais
    if delta == 0:
        return f"A equação possui uma raiz real: x = {x1}"
    else:
        return f"As raízes da equação são: x1 = {x1} e x2 = {x2}"


# Exemplo de uso:
a = float(input("Digite o coeficiente a: "))
b = float(input("Digite o coeficiente b: "))
c = float(input("Digite o coeficiente c: "))

resultado = bhaskara(a, b, c)
print(resultado)
