# Pascal's Law

# Imports
import math

# Constants

# Functions
def pressure():
    # Inputs
    f_1 = float(input("Digite o valor da força 1 sobre a superfície 1 em N:"))
    a_1 = float(input("Digite o valor da área 1 da superfície 1 em m**2:"))
    a_2 = float(input("Digite o valor da área 2 da superfície 2 em m**2:"))

    # Equation
    f_2=(a_2/a_1)*f_1

    print(f"A força 2 imposta sobre a superfíe 2 é de: {float(f_2)} N/m**2")

def main():
    pressure()

# Output
if __name__ == "__main__":
    main()