# Data frame exercises 2
from unittest.mock import inplace

# Imports
import pandas as pd

health_data = pd.read_csv("dataset_2.csv", header=0, sep=",")
health_data.dropna(axis=0, inplace=True)
health_data["Average_Pulse"] = health_data["Average_Pulse"].astype(float)
health_data["Max_Pulse"] = health_data["Max_Pulse"].astype(float)

print(health_data)
print(health_data.info())
print(health_data.describe())