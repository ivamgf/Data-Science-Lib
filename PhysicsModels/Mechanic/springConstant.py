def calcular_coeficiente_restituicao(velocidade_afastamento, velocidade_aproximacao):
    coeficiente_restituicao = abs(velocidade_afastamento) / abs(velocidade_aproximacao)
    return coeficiente_restituicao

# Exemplo de uso:
velocidade_afastamento = float(input("Digite a velocidade de afastamento (em m/s): "))
velocidade_aproximacao = float(input("Digite a velocidade de aproximação (em m/s): "))

coeficiente_restituicao = calcular_coeficiente_restituicao(velocidade_afastamento, velocidade_aproximacao)
print(f"O coeficiente de restituição é {coeficiente_restituicao}")
