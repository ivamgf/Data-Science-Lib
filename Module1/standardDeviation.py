# Imports
import pandas as pd
import matplotlib.pyplot as plt

# Path
data_path = '../Data/data-health.csv'

# Read file
health_data = pd.read_csv(data_path, header=0, sep=",")

# Descrição dos dados
print(health_data.describe())

# Selecionar uma coluna para calcular os percentis (substitua 'column_ref' pelo nome da coluna que deseja analisar)
column_ref = 'Hours_Work'  # Exemplo: 'peso' ou 'altura'

# Calcular percentis e valores mínimo e máximo
percentis = {
    '25%': health_data[column_ref].quantile(0.25),
    '50%': health_data[column_ref].quantile(0.5),  # Mediana
    '75%': health_data[column_ref].quantile(0.75)
}

# Criar gráfico
plt.figure(figsize=(10, 6))
plt.bar(percentis.keys(), percentis.values(), color='skyblue')
plt.title(f'Análise de Percentis de {column_ref.capitalize()}')
plt.ylabel('Valor')
plt.xlabel('Percentis')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Exibir gráfico
plt.show()
