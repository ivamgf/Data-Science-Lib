# Density

# Imports
import math

# Constants

# Functions
def density():
    # Inputs
    m = float(input("Digite a massa da partícula em kg:"))
    volume = float(input("Digite o volume do corpo (em m**3):"))

    # Equation
    density = m/volume

    print(f"A densidade do corpo é de: {float(density)} kg/m**3")

def main():
    density()

# Output
if __name__ == "__main__":
    main()