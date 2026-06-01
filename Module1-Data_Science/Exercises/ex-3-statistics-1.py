import pandas as pd
import numpy as np

health_data = pd.read_csv(
    "dataset_2.csv",
    header=0,
    sep=","
)

max_pulse = health_data["Max_Pulse"].astype(float)

percentile25 = max_pulse.quantile(0.25)
percentile75 = max_pulse.quantile(0.75)

std = np.std(max_pulse)
avg = np.mean(max_pulse)
cv = std/avg
var = np.var(max_pulse)

print(f"Percentil 25 (Q1): {percentile25:.2f}")
print(f"Percentil 75 (Q3): {percentile75:.2f}")
print(f"Desvio Padrão: {std:.2f}")
print(f"Média: {avg:.2f}")
print(f"Coeficiente de variação: {cv:.2f}")
print(f"Variância: {var:.2f}")
