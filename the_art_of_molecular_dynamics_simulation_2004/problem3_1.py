import numpy as np

# Simulation parameters
L = 10.0  # Box size
N_particles = 100  # Number of particles
r_c = 1.0  # Cutoff distance
l_c = r_c  # Cell size
N_cells = int(L / l_c)
delta_r = 0.1  # Maximum particle displacement per time step

# Initialize particle positions randomly
positions = np.random.rand(N_particles, 2) * L

# Initialize cells
cells = [[[] for _ in range(N_cells)] for _ in range(N_cells)]

def assign_particles_to_cells(positions):
    cells = [[[] for _ in range(N_cells)] for _ in range(N_cells)]
    for idx, (x, y) in enumerate(positions):
        i = int(x / l_c) % N_cells
        j = int(y / l_c) % N_cells
        cells[i][j].append(idx)
    return cells

def build_neighbor_lists(cells, positions):
    neighbor_lists = [[] for _ in range(N_particles)]
    for i in range(N_cells):
        for j in range(N_cells):
            # List of particles in the current cell
            current_cell_particles = cells[i][j]
            # Neighboring cells (including periodic boundaries)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni = (i + di) % N_cells
                    nj = (j + dj) % N_cells
                    neighbor_cell_particles = cells[ni][nj]
                    # Compute interactions
                    for idx in current_cell_particles:
                        for jdx in neighbor_cell_particles:
                            if idx < jdx:
                                rij = positions[jdx] - positions[idx]
                                # Apply minimum image convention
                                rij -= L * np.round(rij / L)
                                distance = np.linalg.norm(rij)
                                if distance < r_c:
                                    neighbor_lists[idx].append(jdx)
                                    neighbor_lists[jdx].append(idx)
    return neighbor_lists

# Initial assignment
cells = assign_particles_to_cells(positions)
neighbor_lists = build_neighbor_lists(cells, positions)

# Simulation loop
num_steps = 100
for step in range(num_steps):
    # Update positions (simplified)
    positions += (np.random.rand(N_particles, 2) - 0.5) * delta_r
    positions %= L  # Periodic boundaries

    # Reassign particles to cells
    cells = assign_particles_to_cells(positions)

    # Periodically update neighbor lists
    if step % 10 == 0:
        neighbor_lists = build_neighbor_lists(cells, positions)

    # Compute forces using neighbor lists (not implemented here)
    # forces = compute_forces(positions, neighbor_lists)

    # ... rest of the MD simulation ...

