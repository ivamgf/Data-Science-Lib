import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Reading data
health_data = pd.read_csv("dataset_2.csv", header=0, sep=",")
health_data.dropna(axis=0, inplace=True)

x = health_data["Average_Pulse"]
y = health_data["Calorie_Burnage"]

# Linear regression
slope, intercept, r, p, std_err = linregress(x, y)

def regLin(x):
    return slope * x + intercept

def Predict_Calorie_Burnage(Average_Pulse):
 return(0.3296*Average_Pulse + 346.8662)

myModel = list(map(regLin, x))

# Plot
plt.scatter(x, y)
plt.plot(x, myModel)
plt.ylim(ymin=0, ymax=2000)
plt.xlim(xmin=0, xmax=200)
plt.xlabel("Average_Pulse")
plt.ylabel("Calorie_Burnage")
plt.show()

# Results

# Beta
print("Slope:", slope)
# Alfa
print("Intercept:", intercept)
print("R:", r)
print("R²:", r**2)
print("P-value:", p)
print("Standard Error:", std_err)

print(Predict_Calorie_Burnage(120))
print(Predict_Calorie_Burnage(130))
print(Predict_Calorie_Burnage(150))
print(Predict_Calorie_Burnage(180))