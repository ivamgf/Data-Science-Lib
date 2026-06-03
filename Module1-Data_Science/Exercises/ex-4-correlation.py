# Correlation

# Imports
import pandas as pd
import numpy as np
from unittest.mock import inplace
import matplotlib.pyplot as plt

# Reading data
health_data = pd.read_csv("dataset_2.csv", header=0, sep=",")
health_data.dropna(axis=0, inplace=True)

max_pulse = health_data["Max_Pulse"].astype(float)

Corr_Matrix = round(health_data.corr(),2)

health_data.plot(x='Average_Pulse', y='Calorie_Burnage', kind='scatter')
plt.show()

health_data.info()
health_data.describe()

# Results
print(health_data)
print(max_pulse)
print(Corr_Matrix)
print()