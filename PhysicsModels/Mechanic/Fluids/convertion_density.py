# Convertion Density

# Imports
import math

# Constants
DENSITY_CONVERTION = 1000

# Functions
def density():
    # Inputs
    m = float(input("Digite a massa da partícula em g:"))
    volume = float(input("Digite o volume do corpo (em cm**3):"))

    # Equation
    density = (m/volume) * DENSITY_CONVERTION

    print(f"A densidade do corpo é de: {float(density)} kg/m**3")

def main():
    density()

# Output
if __name__ == "__main__":
    main()