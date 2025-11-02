import math

def calcular_trabalho(forca, distancia, angulo):
    # Convertendo o ângulo de graus para radianos
    angulo_rad = math.radians(angulo)
    trabalho = forca * distancia * math.cos(angulo_rad)
    return trabalho

# Exemplo de uso:
forca = float(input("Digite a força aplicada (em Newtons): "))
distancia = float(input("Digite a distância percorrida (em metros): "))
angulo = float(input("Digite o ângulo entre a força e a direção do movimento (em graus): "))

trabalho = calcular_trabalho(forca, distancia, angulo)
print(f"O trabalho realizado é {trabalho} Joules")
