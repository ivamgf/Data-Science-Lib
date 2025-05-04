# Massa de probabilidade

def funcao_massa_probabilidade(valores_possiveis, probabilidades):
    """
    Cria uma função massa de probabilidade (FMP).
    :param valores_possiveis: lista com os valores que a variável aleatória pode assumir
    :param probabilidades: lista com as probabilidades correspondentes
    :return: dicionário representando a FMP
    """
    if len(valores_possiveis) != len(probabilidades):
        raise ValueError("Listas devem ter o mesmo tamanho.")
    if not abs(sum(probabilidades) - 1.0) < 1e-6:
        raise ValueError("A soma das probabilidades deve ser igual a 1.")

    fmp = dict(zip(valores_possiveis, probabilidades))
    return fmp


def calcular_probabilidade(fmp, x):
    """
    Calcula P(X = x) a partir da função massa de probabilidade.
    :param fmp: dicionário da FMP
    :param x: valor da variável aleatória
    :return: probabilidade associada a x
    """
    return fmp.get(x, 0)  # retorna 0 se x não estiver na FMP


# Exemplo: FMP de um dado justo
valores = [1, 2, 3, 4, 5, 6]
probs = [1 / 6] * 6

fmp_dado = funcao_massa_probabilidade(valores, probs)

# Calcular P(X = 4)
x = 4
print(f"P(X = {x}) = {calcular_probabilidade(fmp_dado, x):.4f}")
