import pandas as pd

# Data
d = {'col1': [1,2,3,4,7], 'col2': [4,5,6,9,5], 'col3' : [7,8,12,1,11]}

# Create Dataframe
df = pd.DataFrame(data=d)

count_column = df.shape[1]
count_row = df.shape[0]

# Show table
print(df)

# Count columns
print('Number of columns:')
print(count_column)

# Count rows
print('Number of rows:')
print(count_row)
