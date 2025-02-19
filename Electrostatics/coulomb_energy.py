#given a list of charges and their positions, calculate the coulomb energy

import numpy as np
import matplotlib.pyplot as plt

# Given charges and their positions
charges = [1, -1, 1, -1]  # in units of elementary charge
positions = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]])  # in arbitrary units

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
T = 300  # Temperature in Kelvin
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
e = 1.602176634e-19  # Elementary charge in Coulombs
k = 1/(4*np.pi*epsilon_0)  # Coulomb constant

# Calculate the Bjerrum length (length at which electrostatic energy equals thermal energy)
l_B = k * e**2 / (k_B * T)

# Calculate Coulomb energy using pair energy summation
def calculate_pair_energy():
    energy = 0
    for i in range(len(charges)):
        for j in range(i+1, len(charges)):
            r_ij = positions[j] - positions[i]
            distance = np.linalg.norm(r_ij)
            if distance > 0:  # Avoid self-interaction
                energy += k * charges[i] * charges[j] * e**2 / distance
    return energy

# Calculate Coulomb energy using potential summation
def calculate_potential_energy():
    energy = 0
    potentials = np.zeros(len(charges))
    
    # First calculate potential at each point due to all other charges
    for i in range(len(charges)):
        for j in range(len(charges)):
            if i != j:
                r_ij = positions[j] - positions[i]
                distance = np.linalg.norm(r_ij)
                potentials[i] += k * charges[j] * e / distance
    
    # Calculate total energy as sum of charge times potential
    for i in range(len(charges)):
        energy += 0.5 * charges[i] * e * potentials[i]
    
    return energy

# Calculate energies using both methods
pair_energy = calculate_pair_energy()
potential_energy = calculate_potential_energy()

print(f"Coulomb energy (pair summation): {pair_energy:.2e} Joules")
print(f"Coulomb energy (potential summation): {potential_energy:.2e} Joules")
print(f"Bjerrum length: {l_B:.2e} meters")

# Visualize the charge configuration
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot charges
for i, (pos, q) in enumerate(zip(positions, charges)):
    color = 'red' if q > 0 else 'blue'
    ax.scatter(*pos, c=color, s=100, label=f'q{i+1}={q}e')

# Add electric field lines (simplified visualization)
x, y, z = np.meshgrid(np.linspace(-0.5, 1.5, 10),
                     np.linspace(-0.5, 1.5, 10),
                     np.linspace(-0.5, 1.5, 10))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Charge Configuration')
plt.legend()
plt.show()


