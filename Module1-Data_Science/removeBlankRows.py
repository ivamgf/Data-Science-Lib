# Imports
import pandas as pd

# Path
data_path = '../Data/data-health.csv'

# Read file
health_data = pd.read_csv(data_path, header=0, sep=",")

health_data.dropna(axis=0,inplace=True)

# Show table
print(health_data)

# Info about data types
print(health_data.info())

# Describe
print(health_data.describe())