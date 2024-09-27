# Imports
import pandas as pd

# Path
data_path = 'Data/data-health.csv'

# Read file
health_data = pd.read_csv(data_path, header=0, sep=",")

# Show table
print(health_data.head())