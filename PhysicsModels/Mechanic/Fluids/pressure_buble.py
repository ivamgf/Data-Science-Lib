# Pressure inside the bubble

# Imports
import math

# Constants


# Functions
def pressure_buble():
    # Inputs
    y = float(input("Digite o valor da tensão superficial em N/M:"))
    r = float(input("Digite o valor do raio em m:"))

    # Equation
    p=(4*y)/r

    print(f"A pressão dentro da bolha é de: {float(p)} N/M**2")

def main():
    pressure_buble()

# Output
if __name__ == "__main__":
    main()