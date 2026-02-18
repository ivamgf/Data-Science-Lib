# Stokes Law

# Imports
import math

# Constants


# Functions
def force():
    # Inputs
    n = float(input("Digite o valor de viscosidade em N.s/m**2:"))
    r = float(input("Digite o valor do raio da esfera que se move no fluído em m):"))
    v = float(input("Digite o valor da velocidade da esfera no fluído em m/s**2:"))

    # Equation
    f = 6*(math.pi)*n*r*v

    print(f"A força é de: {float(f)} N")

def main():
    force()

# Output
if __name__ == "__main__":
    main()