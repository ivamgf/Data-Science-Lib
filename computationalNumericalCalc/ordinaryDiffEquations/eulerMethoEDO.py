import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

# Parameters
# EDO
f = lambda t, s: t**2 + 3

# size
h = 0.1

# Time discret
t = np.arange(0, 1 + h, h)

# Initial condition
s0 = 3

# Euler Method
s = np.zeros(len(t))
s[0] = s0

for i in range(0, len(t) -1):
    s[i + 1] = s[i] + h*f(t[i], s[i])

# Graphics
# Plotting the results of the Euler Method
plt.figure(figsize=(8, 6))

# Plot the numerical solution (Euler Method)
plt.plot(t, s, label='Euler Method', marker='o', linestyle='-', color='cyan')

# Exact solution for comparison (optional)
exact_solution = -np.exp(-t)  # Analytical solution
plt.plot(t, exact_solution, label='Exact Solution', linestyle='--', color='yellow')

# Add labels and title
plt.xlabel('Time (t)')
plt.ylabel('s(t)')
plt.title('Numerical Solution of EDO with Euler Method')

# Add grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Show the plot
plt.show()
