import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

class BondStretchingSimulation:
    def __init__(self):
        # Simulation parameters
        self.k_b = 50.0   # Force constant (reduced for better visualization)
        self.r_0 = 2.0    # Equilibrium bond length
        self.dt = 0.02    # Time step
        self.mass = 1.0   # Mass of the moving atom
        
        # Initial conditions
        self.r = 2.5      # Initial position (stretched)
        self.v = 0.0      # Initial velocity
        
        # Calculate maximum possible energy based on initial stretch
        self.max_stretch = abs(self.r - self.r_0)
        self.max_energy = 0.5 * self.k_b * (self.max_stretch * 2)**2  # Account for both compression and stretching
        
        # Lists to store history
        self.time_points = []
        self.positions = []
        self.energies = []
        
    def calculate_force(self, r):
        """Calculate the restoring force using Hooke's law"""
        return -self.k_b * (r - self.r_0)
    
    def calculate_energy(self, r):
        """Calculate the potential energy"""
        return 0.5 * self.k_b * (r - self.r_0)**2
    
    def update(self):
        """Update the position and velocity using Verlet integration"""
        force = self.calculate_force(self.r)
        acceleration = force / self.mass
        
        # Update position and velocity using Verlet algorithm
        r_new = self.r + self.v * self.dt + 0.5 * acceleration * self.dt**2
        force_new = self.calculate_force(r_new)
        acceleration_new = force_new / self.mass
        v_new = self.v + 0.5 * (acceleration + acceleration_new) * self.dt
        
        self.r = r_new
        self.v = v_new
        
        energy = self.calculate_energy(self.r)
        return self.r, energy

class BondStretchingVisualization:
    def __init__(self):
        self.simulation = BondStretchingSimulation()
        
        # Setup the figure and animation
        plt.style.use('bmh')  # Using bmh style instead of seaborn
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.patch.set_facecolor('white')  # Set figure background to white
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
        
        # Atoms visualization subplot
        self.ax1 = plt.subplot(gs[0])
        self.ax1.set_xlim(-4, 4)
        self.ax1.set_ylim(-4, 4)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Bond Stretching Simulation', pad=20, fontsize=12, fontweight='bold')
        self.ax1.set_facecolor('#f8f9fa')  # Light gray background
        
        # Add equilibrium position marker
        self.ax1.axvline(x=self.simulation.r_0, color='gray', linestyle='--', alpha=0.5)
        self.ax1.text(self.simulation.r_0, 3.5, 'Equilibrium\nPosition', ha='center', va='top')
        
        # Energy plot subplot
        self.ax2 = plt.subplot(gs[1])
        self.ax2.set_xlim(0, 10)
        # Set y-axis limit based on maximum possible energy
        self.ax2.set_ylim(0, self.simulation.max_energy * 1.1)  # Add 10% margin
        self.ax2.set_xlabel('Time (arbitrary units)', fontsize=10)
        self.ax2.set_ylabel('Potential Energy', fontsize=10)
        self.ax2.set_title('Energy vs Time', pad=20, fontsize=12, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('#f8f9fa')  # Light gray background
        
        # Initialize visualization elements
        # Center the first atom at origin
        self.atom1 = Circle((0, 0), 0.3, color='#1f77b4', label='Fixed Atom')  # More vibrant blue
        self.atom2 = Circle((self.simulation.r, 0), 0.3, color='#d62728', label='Moving Atom')  # More vibrant red
        self.bond_line, = self.ax1.plot([], [], 'k-', linewidth=2, label='Bond')
        self.ax1.add_patch(self.atom1)
        self.ax1.add_patch(self.atom2)
        self.energy_line, = self.ax2.plot([], [], '#2ca02c', linewidth=2, label='Potential Energy')  # More vibrant green
        
        # Add legend
        self.ax1.legend(loc='upper left', framealpha=0.9)
        self.ax2.legend(loc='upper right', framealpha=0.9)
        
        # Text displays for current values
        self.text_display = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes,
                                        verticalalignment='top', fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='white', 
                                                 edgecolor='gray', alpha=0.9))
        
        # Data storage
        self.t_data = []
        self.e_data = []
        self.max_energy_seen = 0  # Track maximum energy for verification
        
        # Adjust layout
        plt.tight_layout()
        
    def init(self):
        """Initialize animation"""
        self.bond_line.set_data([], [])
        self.energy_line.set_data([], [])
        return self.atom1, self.atom2, self.bond_line, self.energy_line
        
    def animate(self, frame):
        """Animation update function"""
        # Update simulation
        r, energy = self.simulation.update()
        
        # Track maximum energy for verification
        self.max_energy_seen = max(self.max_energy_seen, energy)
        
        # Update atom position and bond
        self.atom2.center = (r, 0)
        self.bond_line.set_data([0, r], [0, 0])
        
        # Update energy plot
        self.t_data.append(frame * self.simulation.dt)
        self.e_data.append(energy)
        self.energy_line.set_data(self.t_data, self.e_data)
        
        # Update text display
        status_text = (f'Bond Length: {r:.2f}\n'
                      f'Equilibrium: {self.simulation.r_0:.2f}\n'
                      f'Energy: {energy:.2f}\n'
                      f'Max Energy: {self.max_energy_seen:.2f}')
        self.text_display.set_text(status_text)
        
        return self.atom1, self.atom2, self.bond_line, self.energy_line, self.text_display
        
    def run(self):
        """Run the animation"""
        anim = FuncAnimation(self.fig, self.animate, init_func=self.init,
                           frames=1000, interval=20, blit=True)
        plt.show()

if __name__ == "__main__":
    visualization = BondStretchingVisualization()
    visualization.run()
