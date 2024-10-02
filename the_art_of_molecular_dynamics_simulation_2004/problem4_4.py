## Exercise 4.4: Hexatic Phase in Two-Dimensional Liquids

import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
num_particles = 500
box_length = 10.0
positions = np.random.uniform(0, box_length, (num_particles, 2))

# Function to compute nearest neighbors using a cutoff
def compute_neighbors(positions, cutoff):
    num_particles = len(positions)
    neighbor_list = [[] for _ in range(num_particles)]
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            rij = positions[j] - positions[i]
            rij = rij - box_length * np.round(rij / box_length)
            r = np.linalg.norm(rij)
            if r < cutoff:
                neighbor_list[i].append(j)
                neighbor_list[j].append(i)
    return neighbor_list

# Compute the bond-orientational order parameter ψ6
def compute_psi6(positions, neighbor_list):
    psi6 = np.zeros(len(positions), dtype=complex)
    for i in range(len(positions)):
        neighbors = neighbor_list[i]
        N_bonds = len(neighbors)
        if N_bonds == 0:
            continue
        theta_sum = 0
        for j in neighbors:
            rij = positions[j] - positions[i]
            rij = rij - box_length * np.round(rij / box_length)
            theta = np.arctan2(rij[1], rij[0])
            theta_sum += np.exp(1j * 6 * theta)
        psi6[i] = theta_sum / N_bonds
    return psi6

# Parameters
cutoff = 1.5  # Adjust based on density

# Compute neighbors and ψ6
neighbor_list = compute_neighbors(positions, cutoff)
psi6 = compute_psi6(positions, neighbor_list)

# Plot histogram of |ψ6|
print(psi6)
plt.hist(np.abs(psi6), bins=50)
plt.xlabel('|ψ₆|')
plt.ylabel('Frequency')
plt.title('Distribution of Bond-Orientational Order Parameter')
plt.show()
