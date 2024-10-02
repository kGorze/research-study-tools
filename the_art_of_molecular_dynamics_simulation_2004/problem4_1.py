## Exercise 4.1: Comparing Specific Heats

import numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)

# Simulation parameters
temperatures = np.linspace(100, 500, 5)  # Temperatures in Kelvin
num_particles = 500  # Number of particles
num_steps = 10000  # Number of simulation steps
dt = 1e-15  # Time step in seconds

# Placeholder arrays to store results
average_energies = []
cv_fluctuations = []

# Function to perform a simplified MD simulation and return energy data
def simulate_md(T):
    # Initialize kinetic energy array
    kinetic_energies = np.zeros(num_steps)
    # Simplified simulation: kinetic energy fluctuates around average
    avg_kinetic_energy = (3/2) * num_particles * k_B * T
    fluctuations = np.random.normal(0, avg_kinetic_energy * 0.05, num_steps)
    kinetic_energies = avg_kinetic_energy + fluctuations
    return kinetic_energies

# Simulate at different temperatures
for T in temperatures:
    kinetic_energies = simulate_md(T)
    avg_E = np.mean(kinetic_energies)
    avg_E2 = np.mean(kinetic_energies**2)
    Cv_fluct = (avg_E2 - avg_E**2) / (k_B * T**2)
    average_energies.append(avg_E)
    cv_fluctuations.append(Cv_fluct)

# Calculate Cv via derivative dE/dT
average_energies = np.array(average_energies)
cv_derivative = np.gradient(average_energies, temperatures)

# Plotting the results
print(cv_fluctuations)
plt.plot(temperatures, cv_fluctuations, 'o-', label='Cv from fluctuations')
plt.plot(temperatures, cv_derivative, 's--', label='Cv from dE/dT')
plt.xlabel('Temperature (K)')
plt.ylabel('Specific Heat (J/K)')
plt.legend()
plt.title('Comparison of Specific Heat Calculations')
plt.show()
