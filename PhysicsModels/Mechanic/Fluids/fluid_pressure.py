# Fluid Pressure

# Imports
import math

# Constants
PASCAL = 1 # N/m**2 (1 Pascal)
ATM = 1.013 * 10**5 # PA (1 Atmosfera)

# Functions
def density():
    # Inputs
    f_n = float(input("Digite força normal em N:"))
    area = float(input("Digite a área (em m**2):"))

    # Equation
    p=f_n/area

    print(f"A pressão é de: {float(p)} N/m**2")

def main():
    density()

# Output
if __name__ == "__main__":
    main()