import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
num_particles = 100000
num_bins = 100
time_steps = 50
delta_v = 1  # Adjust based on the velocity range

# Initialize velocities (non-equilibrium state)
v_initial = 500  # Arbitrary initial speed (m/s)
speeds = np.full(num_particles, v_initial, dtype=np.float64)  # Ensure speeds is float64

H_values = []

for t in range(time_steps):
    # Simulate particle collisions (simplified for example)
    speeds += np.random.normal(0, 2, num_particles)  # Random perturbation
    speeds = np.abs(speeds)  # Speeds should be positive

    # Compute velocity distribution
    hist, bin_edges = np.histogram(speeds, bins=num_bins, density=True)
    f_v_t = hist + 1e-12  # Add small number to avoid log(0)
    f_v_t /= np.sum(f_v_t * delta_v)  # Normalize the distribution

    # Compute H(t)
    H_t = np.sum(f_v_t * np.log(f_v_t) * delta_v)
    H_values.append(H_t)

# Plot H(t) over time
plt.figure(figsize=(8,6))
plt.plot(range(time_steps), H_values)
plt.xlabel('Time Steps')
plt.ylabel('H(t)')
plt.title('Evolution of H-function Over Time')
plt.show()
