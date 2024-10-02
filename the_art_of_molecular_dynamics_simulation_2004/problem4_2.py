## Exercise 4.2: Errors Due to Interaction Cutoff

import numpy as np
import matplotlib.pyplot as plt

# LJ potential function with cutoff
def lj_potential(r, epsilon, sigma, cutoff):
    if r < cutoff:
        return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    else:
        return 0.0

# Simulation parameters
epsilon = 1.0  # Depth of the potential well
sigma = 1.0    # Finite distance where the potential is zero
cutoff_distances = [2.5, 3.0, 4.0, 5.0]  # Cutoff distances in units of sigma
num_particles = 100
box_length = 10.0  # Simulation box length

# Generate random positions for particles
np.random.seed(42)
positions = np.random.uniform(0, box_length, (num_particles, 3))

# Function to calculate total potential energy for a given cutoff
def calculate_total_energy(positions, cutoff):
    total_energy = 0.0
    num_particles = len(positions)
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            rij = positions[j] - positions[i]
            rij = rij - box_length * np.round(rij / box_length)  # Periodic boundaries
            r = np.linalg.norm(rij)
            energy = lj_potential(r, epsilon, sigma, cutoff)
            total_energy += energy
    return total_energy

# Calculate energies for different cutoffs
energies = []
for cutoff in cutoff_distances:
    energy = calculate_total_energy(positions, cutoff)
    energies.append(energy)

# Plotting the total energy vs. cutoff distance
print(energies)
plt.plot(cutoff_distances, energies, 'o-')
plt.xlabel('Cutoff Distance (σ)')
plt.ylabel('Total Potential Energy')
plt.title('Effect of Cutoff Distance on Total Energy')
plt.show()
