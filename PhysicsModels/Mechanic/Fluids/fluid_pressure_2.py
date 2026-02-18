# Pressure of a fluid with constant density

# Imports
import math

# Constants
PASCAL = 1 # N/m**2 (1 Pascal)
ATM = 1.013 * 10**5 # PA (1 Atmosfera)

# Functions
def pressure():
    # Inputs
    g = 9.8 # m/s**2
    pressure_initial = float(input("Digite pressão na superfície do fluído em N/m**2:"))
    density = float(input("Digite a densidade (em kg/m**3):"))
    h = float(input("Digite a altura (em m):"))

    # Equation
    p=pressure_initial + (density * g * h)

    print(f"A pressão no fluído é de: {float(p)} N/m**2")

def main():
    pressure()

# Output
if __name__ == "__main__":
    main()