## Exercise 4.5: RDF Differences Between LJ and Soft-Spheres

import numpy as np
import matplotlib.pyplot as plt

# Function to compute RDF
def compute_rdf(positions, box_length, dr, max_r):
    num_particles = len(positions)
    rdf_bins = np.arange(0, max_r, dr)
    rdf = np.zeros(len(rdf_bins) - 1)
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            rij = positions[j] - positions[i]
            rij = rij - box_length * np.round(rij / box_length)
            r = np.linalg.norm(rij)
            if r < max_r:
                bin_index = int(r / dr)
                if bin_index < len(rdf):  # Ensure bin_index is within bounds
                    rdf[bin_index] += 2  # Account for both i->j and j->i
    # Normalize RDF
    density = num_particles / box_length**3
    shell_volumes = (4/3) * np.pi * ((rdf_bins[1:]**3) - (rdf_bins[:-1]**3))
    rdf = rdf / (density * num_particles * shell_volumes)
    r_values = (rdf_bins[1:] + rdf_bins[:-1]) / 2
    return r_values, rdf


# Simulation parameters
num_particles = 500
box_length = 10.0
positions_LJ = np.random.uniform(0, box_length, (num_particles, 3))
positions_SS = np.random.uniform(0, box_length, (num_particles, 3))

# Placeholder for positions; in practice, obtain from simulations
# Compute RDFs
dr = 0.05
max_r = box_length / 2

r_LJ, rdf_LJ = compute_rdf(positions_LJ, box_length, dr, max_r)
r_SS, rdf_SS = compute_rdf(positions_SS, box_length, dr, max_r)

# Plotting
print(r_LJ)
plt.plot(r_LJ, rdf_LJ, label='Lennard-Jones')
plt.plot(r_SS, rdf_SS, label='Soft-Sphere')
plt.xlabel('Distance r')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function Comparison')
plt.legend()
plt.show()
