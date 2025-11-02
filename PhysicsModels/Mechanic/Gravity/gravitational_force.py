# Gravitational Force

# Constants
G = 6.67259 * (10**(-11)) # N.m**2 / kg**2

# Functions
def gravitational_force():
    # Inputs
    m1 = float(input("Digite a massa da partícula 1 em kg:"))
    m2 = float(input("Digite a massa da partícula 2 em kg:"))
    r = float(input("Digite o valor do raio em metros:"))

    # Equations
    f = G*((m1*m2)/(r**2))

    print(f"A força gravitacional entre as 2 partículas é de: {float(f)} N.m**2 / kg**2")

def main():
    gravitational_force()

# Output
if __name__ == "__main__":
    main()

