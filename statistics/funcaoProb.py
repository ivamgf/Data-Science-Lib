# Função de Probabilidade

def criar_variavel():
    """
    Lê os valores e probabilidades do usuário para uma única variável aleatória.
    """
    valores = input("Digite os valores possíveis separados por espaço (ex: 0 1 2): ").split()
    valores = [int(v) for v in valores]

    probabilidades = input("Digite as probabilidades correspondentes (ex: 0.2 0.5 0.3): ").split()
    probabilidades = [float(p) for p in probabilidades]

    if len(valores) != len(probabilidades):
        raise ValueError("Erro: número de valores e probabilidades deve ser igual.")
    if not abs(sum(probabilidades) - 1.0) < 1e-6:
        raise ValueError("Erro: a soma das probabilidades deve ser 1.")

    return dict(zip(valores, probabilidades))


def calcular_probabilidade(fmp, valor_desejado):
    """
    Calcula a probabilidade de um único valor em uma variável aleatória.
    """
    return fmp.get(valor_desejado, 0)


# 🌟 Execução principal
def main():
    print("📊 Criando uma variável aleatória discreta")
    fmp = criar_variavel()

    valor = int(input("\nDigite o valor desejado para calcular a probabilidade: "))
    prob = calcular_probabilidade(fmp, valor)

    print(f"\nP(X = {valor}) = {prob:.4f}")


if __name__ == "__main__":
    main()
