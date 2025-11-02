# Orbital Period

# Imports
import math

# Constants
G = 6.67259 * (10**(-11)) # N.m**2 / kg**2
M_T = 5.974 * 10**24 # kg (Massa da terra)
R_T = 6.38 * 10**6 # metros (Raio da terra)

# Functions
def period():
    # Inputs
    # m = float(input("Digite a massa do planeta (ou outro corpo celeste) em kg:"))
    r = float(input("Digite a altitude do satélite (em metros):"))

    # Equations
    v = (2*math.pi*(math.sqrt(r**3)))/(math.sqrt(G*M_T))

    print(f"O período de órbita de um satélite será de: {float(v)} m/s")

def main():
    period()

# Output
if __name__ == "__main__":
    main()