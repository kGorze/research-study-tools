import numpy as np

# Define potential parameters
epsilon = 1.0
sigma = 1.0
r_min = 0.9 * sigma
r_max = 2.5 * sigma
delta_r = 0.001 * sigma
r_values = np.arange(r_min, r_max, delta_r)

# Precompute V(r) and F(r)
V_table = 4 * epsilon * ((sigma / r_values)**12 - (sigma / r_values)**6)
F_table = 24 * epsilon / r_values * (2 * (sigma / r_values)**12 - (sigma / r_values)**6)

# Function to get F(r) using interpolation
def get_force(r):
    if r < r_min or r >= r_max:
        return 0.0
    index = int((r - r_min) / delta_r)
    r_i = r_values[index]
    F_i = F_table[index]
    F_ip1 = F_table[index + 1]
    # Linear interpolation
    F = F_i + (F_ip1 - F_i) * (r - r_i) / delta_r
    return F

# Example usage in force computation
def compute_forces(positions):
    forces = np.zeros_like(positions)
    N = len(positions)
    for i in range(N):
        for j in range(i+1, N):
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)
            F = get_force(r)
            fij = F * rij / r
            forces[i] += fij
            forces[j] -= fij
    return forces
