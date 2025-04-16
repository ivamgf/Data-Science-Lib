# Alocação ótima de Newman

import math

def calcular_tamanho_amostra_neyman(Nh_list, Sh_list, z, E):
    """
    Calcula o tamanho mínimo da amostra total com alocação ótima de Neyman.
    """
    Nh_Sh = [Nh * Sh for Nh, Sh in zip(Nh_list, Sh_list)]
    Nh_Sh2 = [Nh * (Sh ** 2) for Nh, Sh in zip(Nh_list, Sh_list)]

    soma_Nh = sum(Nh_list)
    soma_Nh_Sh = sum(Nh_Sh)
    soma_Nh_Sh2 = sum(Nh_Sh2)

    numerador = (soma_Nh_Sh ** 2) * (z ** 2)
    denominador = (E ** 2) * (soma_Nh ** 2) + (z ** 2) * soma_Nh_Sh2

    n_total = numerador / denominador
    return round(n_total)

def alocacao_otima_neyman(Nh_list, Sh_list, n_total):
    """
    Aloca a amostra total entre os estratos com a alocação ótima de Neyman.
    """
    total_peso = sum(Nh * Sh for Nh, Sh in zip(Nh_list, Sh_list))
    nh_list = [(Nh * Sh / total_peso) * n_total for Nh, Sh in zip(Nh_list, Sh_list)]
    nh_list = [round(nh) for nh in nh_list]
    return nh_list

# Valores fornecidos
Nh = [90, 60, 30]                      # Tamanho dos estratos
variancias = [40000, 22500, 10000]     # Variâncias dos estratos
Sh = [math.sqrt(v) for v in variancias]  # Desvios padrão

z = 1.96   # Nível de confiança de 95%
E = 0.03   # Erro amostral de 3%

# Cálculo do tamanho da amostra total
n_total = calcular_tamanho_amostra_neyman(Nh, Sh, z, E)
print(f"\nTamanho total mínimo da amostra: {n_total}")

# Alocação ótima de Neyman
amostras_por_estrato = alocacao_otima_neyman(Nh, Sh, n_total)

# Impressão do resultado
for i, nh in enumerate(amostras_por_estrato):
    print(f"Estrato {i+1}: {nh} elementos")
