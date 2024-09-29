# Imports
import pandas as pd

# Path
data_path = '../Data/data-health.csv'

# Read file
health_data = pd.read_csv(data_path, header=0, sep=",")

# Describe
print(health_data.describe())