# Imports
import pandas as pd
import numpy as np

# Data
data = {
    'Duration': [30,30,45,45,45,60,60,60,75,75],
    'Average_Pulse': [80,85,90,95,100,105,110,115,120,125],
    'Max_Pulse': [120,120,130,130,140,140,145,145,150,150],
    'Calorie_Burnage': [240,250,260,270,280,290,300,310,320,330],
    'Hours_Work': [10,10,8,8,0,7,7,8,0,8],
    'Hours_Sleep': [7,7,7,7,7,8,8,8,8,8]
}

# Create DataFrame
df = pd.DataFrame(data=data)

# Convert columns to arrays NumPy
average_pulse = np.array(df['Average_Pulse'])
calorie_burnage = np.array(df['Calorie_Burnage'])

# Max, Min and Mean Function
average_pulse_max = np.max(average_pulse)
average_pulse_min = np.min(average_pulse)
average_calorie_burnage = np.mean(calorie_burnage)

# Table
print('The Sports Watch Data Set')
print(df)

# Max Function
print('Max value of Average Pulse:')
print(average_pulse_max)

# Min Function
print('Min value of Average Pulse:')
print(average_pulse_min)

# Mean Function
print('Mean value of Calorie Burnage:')
print(average_calorie_burnage)
