# Surface Tension

# Imports
import math

# Constants


# Functions
def surface_tension():
    # Inputs
    f = float(input("Digite a força que atua sobre o corpo em N:"))
    d = float(input("Digite o comprimento total ao longo do qual atua a força em m:"))

    # Equation
    y=f/d

    print(f"A tensão superficial da película é de: {float(y)} N/m")

def main():
    surface_tension()

# Output
if __name__ == "__main__":
    main()