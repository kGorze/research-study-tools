import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
num_particles = 10
num_steps = 200
box_size = 100

# Initialize positions and velocities randomly
positions = np.random.rand(num_particles, 2) * box_size
velocities = np.random.randn(num_particles, 2)

# Store positions for all time steps
all_positions = np.zeros((num_steps, num_particles, 2))

for t in range(num_steps):
    positions += velocities  # Update positions
    # Reflect off walls
    velocities[positions > box_size] *= -1
    velocities[positions < 0] *= -1
    positions = np.clip(positions, 0, box_size)
    all_positions[t] = positions

# Set up the plot
fig, ax = plt.subplots()
scat = ax.scatter(all_positions[0,:,0], all_positions[0,:,1])
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_title('Particle Trajectories')

# Update function for animation
def update(frame):
    scat.set_offsets(all_positions[frame])
    return scat,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=50, blit=True)

plt.show()
