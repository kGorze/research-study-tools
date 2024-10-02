import numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
T = 300  # Temperature (K)
m = 6.6335209e-26  # Mass of an argon atom (kg)

# Generate sample velocities from the simulation (replace with actual simulation data)
num_particles = 100000
np.random.seed(0)
v_x = np.random.normal(0, np.sqrt(k_B*T/m), num_particles)
v_y = np.random.normal(0, np.sqrt(k_B*T/m), num_particles)
v_z = np.random.normal(0, np.sqrt(k_B*T/m), num_particles)
speeds = np.sqrt(v_x**2 + v_y**2 + v_z**2)

# Create histogram of observed speeds
bins = np.linspace(0, np.max(speeds), 100)
hist, bin_edges = np.histogram(speeds, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Theoretical Maxwell-Boltzmann distribution
v = np.linspace(0, np.max(speeds), 1000)
f_v = 4 * np.pi * (m / (2 * np.pi * k_B * T))**(1.5) * v**2 * np.exp(-m * v**2 / (2 * k_B * T))

# Plotting
plt.figure(figsize=(8,6))
plt.plot(bin_centers, hist, label='Observed Distribution', linestyle='dotted')
plt.plot(v, f_v, label='Maxwell-Boltzmann Distribution', color='red')
plt.xlabel('Speed (m/s)')
plt.ylabel('Probability Density')
plt.title('Comparison of Observed and Theoretical Velocity Distributions')
plt.legend()
plt.show()
