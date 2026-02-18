# Viscosity

# Imports
import math

# Constants


# Functions
def viscosity():
    # Inputs
    # Shear stress (Tensão de cisalhamento)
    f = float(input("Digite o valor da força em N:"))
    a = float(input("Digite o valor da área da seção em m**2:"))
    # Strain rate (Taxa de deformação)
    v = float(input("Digite o valor da velocidade de escoamento da superfície em m/s**2:"))
    l = float(input("Digite o valor do comprimento do deslocamento em m:"))

    # Equations
    shear_stress = f/a

    strain_rate = v/l

    viscosity = shear_stress/strain_rate

    print(f"A Tensão de cisalhamento é de: {float(shear_stress)} N/m**2")
    print(f"A Taxa de deformação é de: {float(strain_rate)} s**2")
    print(f"A viscosidade é de: {float(viscosity)} N.s/m**2 = Pa.s")

def main():
    viscosity()

# Output
if __name__ == "__main__":
    main()