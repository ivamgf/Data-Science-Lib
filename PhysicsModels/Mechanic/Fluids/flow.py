# Flow

# Imports
import math

# Constants


# Functions
def flow():
    # Inputs
    a_1 = float(input("Digite área da seção 1 em m**2:"))
    a_2 = float(input("Digite área da seção 2 em m**2:"))
    v_2 = float(input("Digite a velocidade de escoamento da seção 2 em m/s**2):"))

    # Equation
    v_1=(a_2*v_2)/a_1

    print(f"A velocidade de escoamento da seção 1 é de: {float(v_1)} m/s**2")

def main():
    flow()

# Output
if __name__ == "__main__":
    main()