# Distribuição normal

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def main():
    print("### Análise de Distribuição Normal ###")

    # Entrada dos parâmetros
    media = float(input("Digite a média da distribuição: "))
    desvio = float(input("Digite o desvio padrão: "))

    # Intervalo para análise
    x_min = float(input("Valor mínimo de x para o gráfico: "))
    x_max = float(input("Valor máximo de x para o gráfico: "))

    # Ponto específico para cálculo de PDF e CDF
    x_val = float(input("Digite o valor de x para calcular PDF e CDF: "))

    # Geração dos dados
    x = np.linspace(x_min, x_max, 1000)
    pdf = norm.pdf(x, loc=media, scale=desvio)
    cdf = norm.cdf(x, loc=media, scale=desvio)

    # Cálculo dos valores para o ponto específico
    pdf_val = norm.pdf(x_val, loc=media, scale=desvio)
    cdf_val = norm.cdf(x_val, loc=media, scale=desvio)

    print(f"\nPDF em x = {x_val}: {pdf_val:.5f}")
    print(f"CDF em x = {x_val}: {cdf_val:.5f}")

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x, pdf, label='PDF', color='blue')
    plt.axvline(x_val, color='blue', linestyle='--', label=f'PDF(x={x_val})')
    plt.title("Função Densidade de Probabilidade (PDF)")
    plt.xlabel("x")
    plt.ylabel("Densidade")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, cdf, label='CDF', color='green')
    plt.axvline(x_val, color='green', linestyle='--', label=f'CDF(x={x_val})')
    plt.title("Função de Distribuição Acumulada (CDF)")
    plt.xlabel("x")
    plt.ylabel("Probabilidade acumulada")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
