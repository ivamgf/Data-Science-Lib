# Data frame exercises 2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import webbrowser
import os

# ===================================
# CONFIGURAÇÕES DE EXIBIÇÃO
# ===================================

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ===================================
# CARREGAR DADOS
# ===================================

health_data = pd.read_csv(
    "dataset_2.csv",
    header=0,
    sep=","
)

# ===================================
# LIMPEZA DOS DADOS
# ===================================

health_data.dropna(
    axis=0,
    inplace=True
)

health_data["Average_Pulse"] = (
    health_data["Average_Pulse"]
    .astype(float)
)

health_data["Max_Pulse"] = (
    health_data["Max_Pulse"]
    .astype(float)
)

# ===================================
# INFORMAÇÕES DOS DADOS
# ===================================

print("\nINFO")
health_data.info()

# ===================================
# ESTATÍSTICAS DESCRITIVAS
# ===================================

desc = health_data.describe()

print("\nDESCRIBE")
print(desc)

# ===================================
# VARIÁVEL ANALISADA
# ===================================

coluna = "Average_Pulse"

dados = health_data[coluna]

# ===================================
# MEDIDAS ESTATÍSTICAS
# ===================================

media = dados.mean()
mediana = dados.median()

q1 = dados.quantile(0.25)
q3 = dados.quantile(0.75)

minimo = dados.min()
maximo = dados.max()

desvio = dados.std()

# ===================================
# TABELA DE ESTATÍSTICAS
# ===================================

estatisticas = pd.DataFrame({
    "Métrica": [
        "Mínimo",
        "1º Quartil (Q1)",
        "Mediana (Q2)",
        "Média",
        "3º Quartil (Q3)",
        "Máximo",
        "Desvio Padrão"
    ],
    "Valor": [
        minimo,
        q1,
        mediana,
        media,
        q3,
        maximo,
        desvio
    ]
})

print("\nESTATÍSTICAS")
print(estatisticas)

# ===================================
# GRÁFICO
# HISTOGRAMA + CURVA NORMAL
# ===================================

plt.figure(figsize=(14, 8))

# Histograma

plt.hist(
    dados,
    bins=12,
    density=True,
    alpha=0.5,
    edgecolor="black",
    label="Histograma"
)

# Curva normal

x = np.linspace(
    minimo,
    maximo,
    1000
)

y = (
    1 /
    (desvio * np.sqrt(2 * np.pi))
) * np.exp(
    -((x - media) ** 2)
    /
    (2 * desvio ** 2)
)

plt.plot(
    x,
    y,
    linewidth=3,
    label="Distribuição Normal"
)

# ===================================
# LINHAS DAS MEDIDAS ESTATÍSTICAS
# ===================================

plt.axvline(
    minimo,
    linestyle="--",
    linewidth=2,
    label=f"Mínimo = {minimo:.2f}"
)

plt.axvline(
    q1,
    linestyle="--",
    linewidth=2,
    label=f"Q1 = {q1:.2f}"
)

plt.axvline(
    mediana,
    linestyle="-.",
    linewidth=2,
    label=f"Mediana = {mediana:.2f}"
)

plt.axvline(
    media,
    linewidth=3,
    label=f"Média = {media:.2f}"
)

plt.axvline(
    q3,
    linestyle="--",
    linewidth=2,
    label=f"Q3 = {q3:.2f}"
)

plt.axvline(
    maximo,
    linestyle="--",
    linewidth=2,
    label=f"Máximo = {maximo:.2f}"
)

# ===================================
# TEXTO COM ESTATÍSTICAS
# ===================================

texto = (
    f"Média: {media:.2f}\n"
    f"Mediana: {mediana:.2f}\n"
    f"Desvio Padrão: {desvio:.2f}\n"
    f"Q1: {q1:.2f}\n"
    f"Q3: {q3:.2f}\n"
    f"Mínimo: {minimo:.2f}\n"
    f"Máximo: {maximo:.2f}"
)

plt.text(
    0.98,
    0.98,
    texto,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

# ===================================
# FORMATAÇÃO DO GRÁFICO
# ===================================

plt.title(
    f"Distribuição Normal (Gaussiana) - {coluna}",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    coluna,
    fontsize=12
)

plt.ylabel(
    "Densidade",
    fontsize=12
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    loc="upper left"
)

plt.tight_layout()

# ===================================
# SALVAR GRÁFICO
# ===================================

grafico = "gaussiana.png"

plt.savefig(
    grafico,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ===================================
# GERAR RELATÓRIO HTML
# ===================================

with open(
    "resultado.html",
    "w",
    encoding="utf-8"
) as f:

    f.write("<h1>Dataset</h1>")
    f.write(
        health_data.to_html(
            index=False
        )
    )

    f.write("<hr>")

    f.write("<h1>Estatísticas Descritivas</h1>")
    f.write(
        desc.to_html()
    )

    f.write("<hr>")

    f.write("<h1>Medidas Estatísticas</h1>")
    f.write(
        estatisticas.to_html(
            index=False
        )
    )

    f.write("<hr>")

    f.write(
        f"<h1>Distribuição Normal (Gaussiana) - {coluna}</h1>"
    )

    f.write(
        f'<img src="{grafico}" width="1200">'
    )

# ===================================
# ABRIR RELATÓRIO NO NAVEGADOR
# ===================================

arquivo_html = os.path.abspath(
    "resultado.html"
)

webbrowser.open(
    "file://" + arquivo_html
)

print(
    "\nRelatório HTML aberto no navegador."
)