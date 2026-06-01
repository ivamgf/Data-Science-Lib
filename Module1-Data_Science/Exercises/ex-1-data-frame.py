# Data frame exercises

# Imports
import pandas as pd
from pandas import DataFrame

d = {
    'col1': [1, 2, 3, 4, 7],
    'col2': [4, 5, 6, 9, 5],
    'col3': [7, 8, 12, 1, 11]
}

dataframe = pd.DataFrame(data=d)

major = dataframe['col1'].max()
minor = dataframe['col2'].min()
avg = dataframe['col3'].mean()

print(dataframe)
print(major)
print(minor)
print(avg)
