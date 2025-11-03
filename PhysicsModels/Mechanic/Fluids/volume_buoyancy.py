# Volume x Buoyancy

# Imports
import math

# Constants

# Functions
def volume():
    # Inputs
    m = float(input("Digite a massa em kg:"))
    density = float(input("Digite a densidade do corpo (em kg/m**3):"))

    # Equation
    volume=m/density

    print(f"O volume do corpo é de: {float(volume)} m**3")

def main():
    volume()

# Output
if __name__ == "__main__":
    main()