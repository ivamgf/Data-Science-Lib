import matplotlib.pyplot as plt

def calcular_impulso(forca, tempo):
    impulso = forca * tempo
    return impulso

# Exemplo de uso:
forca = float(input("Digite a força aplicada (em Newtons): "))
tempo = float(input("Digite o tempo de aplicação da força (em segundos): "))

# Calcular o impulso
impulso = calcular_impulso(forca, tempo)

# Exibir o valor do impulso
print(f"O impulso é {impulso} N·s")

# Gerar dados para o gráfico: Força constante ao longo do tempo
tempos = [0, tempo]  # Tempo inicial 0 até o tempo inserido
forcas = [forca, forca]  # Força constante durante o tempo

# Plotar o gráfico da Força pelo Tempo
plt.plot(tempos, forcas, label=f'Força = {forca} N')
plt.fill_between(tempos, forcas, alpha=0.3, color='blue')  # Preencher a área sob a curva

# Configurações do gráfico
plt.title('Gráfico da Força pelo Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Força (N)')
plt.legend()

# Exibir o gráfico
plt.grid(True)
plt.show()
