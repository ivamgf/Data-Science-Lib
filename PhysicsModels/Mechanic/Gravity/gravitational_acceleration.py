# Gravitational acceleration

# Constants
G = 6.67259 * (10**(-11)) # N.m**2 / kg**2
M_T = 5.974 * 10**24 # kg (Massa da terra)
R_T = 6.38 * 10**6 # metros (Raio da terra)

# Functions
def acceleration():
    # Inputs
    m_p = float(input("Digite a massa do planeta (ou outro corpo celeste) em kg:"))
    r_p = float(input("Digite o raio do planeta (ou outro corpo celeste em metros):"))

    # Equations
    g = (G*m_p)/(r_p**2)

    print(f"A aceleração gravitacional em um planeta ou outro corpo celeste é de: {float(g)} m/s**2")

def main():
    acceleration()

# Output
if __name__ == "__main__":
    main()