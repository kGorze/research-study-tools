## Exercise 4.3: Soft-Sphere Equation of State Near Melting Transition

import numpy as np
import matplotlib.pyplot as plt

# Soft-sphere potential
def soft_sphere_potential(r, epsilon, sigma, n):
    return epsilon * (sigma / r)**n

# Simulation parameters
epsilon = 1.0
k_B = 1.380649e-23  # Boltzmann constant in J/K

sigma = 1.0
n = 12  # Exponent for the soft-sphere potential
densities = np.linspace(0.8, 1.2, 5)  # Reduced densities
temperature = 1.0  # Reduced temperature

# Function to calculate pressure using the virial theorem
def calculate_pressure(positions, forces, volume):
    virial = np.sum(positions * forces)
    pressure = (num_particles * k_B * temperature + virial) / (3 * volume)
    return pressure

# Placeholder for pressures at different densities
pressures = []

for rho in densities:
    num_particles = 500
    volume = num_particles / rho
    box_length = volume**(1/3)
    positions = np.random.uniform(0, box_length, (num_particles, 3))
    # Placeholder for forces calculation
    # In a full simulation, compute forces and integrate motion
    forces = np.zeros_like(positions)
    # Compute pressure
    pressure = (num_particles * k_B * temperature) / volume  # Ideal gas approximation
    pressures.append(pressure)

# Plotting pressure vs. density
print(pressures)
plt.plot(densities, pressures, 'o-')
plt.xlabel('Density (ρ)')
plt.ylabel('Pressure (P)')
plt.title('Equation of State for Soft-Sphere System')
plt.show()
