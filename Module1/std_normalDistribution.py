# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Path
data_path = '../Data/data-health.csv'

# Read file
health_data = pd.read_csv(data_path, header=0, sep=",")

# Descrição dos dados
print(health_data.describe())

# Selecionar uma coluna para análise (substitua 'Hours_Work' pela coluna que deseja analisar)
column_ref = 'Hours_Work'  # Exemplo: 'peso' ou 'altura'

# Calcular média e desvio padrão para a distribuição normal
mean = health_data[column_ref].mean()
std_dev = health_data[column_ref].std()

# Gerar dados para a distribuição normal
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
y = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

# Criar gráfico
plt.figure(figsize=(10, 6))

# Plotando a distribuição normal
plt.plot(x, y, color='red', label=f'Distribuição Normal\nMédia: {mean:.2f}, Desvio Padrão: {std_dev:.2f}')
plt.fill_between(x, y, color='skyblue', alpha=0.5)

# Plotando o histograma dos dados reais
sns.histplot(health_data[column_ref], kde=True, color='blue', stat='density', bins=20, label='Dados Reais')

# Personalizando o gráfico
plt.title(f'Distribuição Normal de {column_ref.capitalize()}')
plt.xlabel(f'{column_ref.capitalize()}')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)

# Exibir gráfico
plt.show()
