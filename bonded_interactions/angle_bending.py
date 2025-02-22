import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

class AngleBendingSimulation:
    def __init__(self):
        # Simulation parameters
        self.k_theta = 30.0  # Angular force constant
        self.theta_0 = np.pi * 120/180  # Equilibrium angle (120 degrees in radians)
        self.dt = 0.02    # Time step
        self.moment_of_inertia = 1.0  # Moment of inertia for angular motion
        
        # Initial conditions
        self.theta = np.pi * 90/180  # Initial angle (90 degrees in radians)
        self.omega = 0.0  # Initial angular velocity
        
        # Fixed parameters
        self.bond_length = 2.0  # Fixed bond length
        
        # Calculate maximum possible energy based on initial angular displacement
        self.max_angle_deviation = abs(self.theta - self.theta_0)
        self.max_energy = 0.5 * self.k_theta * (self.max_angle_deviation * 2)**2
        
    def calculate_torque(self, theta):
        """Calculate the restoring torque using angular Hooke's law"""
        return -self.k_theta * (theta - self.theta_0)
    
    def calculate_energy(self, theta):
        """Calculate the potential energy"""
        return 0.5 * self.k_theta * (theta - self.theta_0)**2
    
    def update(self):
        """Update angle and angular velocity using Verlet integration"""
        torque = self.calculate_torque(self.theta)
        angular_acceleration = torque / self.moment_of_inertia
        
        # Update angle and angular velocity using Verlet algorithm
        theta_new = self.theta + self.omega * self.dt + 0.5 * angular_acceleration * self.dt**2
        torque_new = self.calculate_torque(theta_new)
        angular_acceleration_new = torque_new / self.moment_of_inertia
        omega_new = self.omega + 0.5 * (angular_acceleration + angular_acceleration_new) * self.dt
        
        self.theta = theta_new
        self.omega = omega_new
        
        energy = self.calculate_energy(self.theta)
        return self.theta, energy

class AngleBendingVisualization:
    def __init__(self):
        self.simulation = AngleBendingSimulation()
        
        # Setup the figure and animation
        plt.style.use('bmh')
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.patch.set_facecolor('white')
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
        
        # Atoms visualization subplot
        self.ax1 = plt.subplot(gs[0])
        self.ax1.set_xlim(-4, 4)
        self.ax1.set_ylim(-4, 4)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Angle Bending Simulation', pad=20, fontsize=12, fontweight='bold')
        self.ax1.set_facecolor('#f8f9fa')
        
        # Add equilibrium angle marker
        self.equilibrium_line = self.plot_equilibrium_angle()
        
        # Energy plot subplot
        self.ax2 = plt.subplot(gs[1])
        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(0, self.simulation.max_energy * 1.1)
        self.ax2.set_xlabel('Time (arbitrary units)', fontsize=10)
        self.ax2.set_ylabel('Potential Energy', fontsize=10)
        self.ax2.set_title('Energy vs Time', pad=20, fontsize=12, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('#f8f9fa')
        
        # Initialize visualization elements
        # Central atom at origin
        self.atom_center = Circle((0, 0), 0.3, color='#1f77b4', label='Central Atom')
        # Left atom fixed
        self.atom_left = Circle((-self.simulation.bond_length, 0), 0.3, color='#2ca02c', label='Fixed Atom')
        # Right atom moving
        x = self.simulation.bond_length * np.cos(self.simulation.theta)
        y = self.simulation.bond_length * np.sin(self.simulation.theta)
        self.atom_right = Circle((x, y), 0.3, color='#d62728', label='Moving Atom')
        
        # Add bonds
        self.bond_left, = self.ax1.plot([], [], 'k-', linewidth=2)
        self.bond_right, = self.ax1.plot([], [], 'k-', linewidth=2)
        
        # Add atoms to plot
        self.ax1.add_patch(self.atom_center)
        self.ax1.add_patch(self.atom_left)
        self.ax1.add_patch(self.atom_right)
        
        # Add energy plot line
        self.energy_line, = self.ax2.plot([], [], '#2ca02c', linewidth=2, label='Potential Energy')
        
        # Add legends
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
        self.max_energy_seen = 0
        
        # Adjust layout
        plt.tight_layout()
    
    def plot_equilibrium_angle(self):
        """Plot the equilibrium angle marker"""
        theta = self.simulation.theta_0
        x = self.simulation.bond_length * np.cos(theta)
        y = self.simulation.bond_length * np.sin(theta)
        line, = self.ax1.plot([0, x], [0, y], '--', color='gray', alpha=0.5)
        self.ax1.text(x * 1.2, y * 1.2, 'Equilibrium\nAngle', ha='center', va='center')
        return line
        
    def init(self):
        """Initialize animation"""
        self.bond_left.set_data([], [])
        self.bond_right.set_data([], [])
        self.energy_line.set_data([], [])
        return self.atom_center, self.atom_left, self.atom_right, self.bond_left, self.bond_right, self.energy_line
        
    def animate(self, frame):
        """Animation update function"""
        # Update simulation
        theta, energy = self.simulation.update()
        
        # Track maximum energy
        self.max_energy_seen = max(self.max_energy_seen, energy)
        
        # Update right atom position and bonds
        x = self.simulation.bond_length * np.cos(theta)
        y = self.simulation.bond_length * np.sin(theta)
        self.atom_right.center = (x, y)
        
        # Update bonds
        self.bond_left.set_data([-self.simulation.bond_length, 0], [0, 0])
        self.bond_right.set_data([0, x], [0, y])
        
        # Update energy plot
        self.t_data.append(frame * self.simulation.dt)
        self.e_data.append(energy)
        self.energy_line.set_data(self.t_data, self.e_data)
        
        # Update text display
        status_text = (f'Angle: {np.degrees(theta):.1f}°\n'
                      f'Equilibrium: {np.degrees(self.simulation.theta_0):.1f}°\n'
                      f'Energy: {energy:.2f}\n'
                      f'Max Energy: {self.max_energy_seen:.2f}')
        self.text_display.set_text(status_text)
        
        return self.atom_center, self.atom_left, self.atom_right, self.bond_left, self.bond_right, self.energy_line, self.text_display
        
    def run(self):
        """Run the animation"""
        anim = FuncAnimation(self.fig, self.animate, init_func=self.init,
                           frames=1000, interval=20, blit=True)
        plt.show()

if __name__ == "__main__":
    visualization = AngleBendingVisualization()
    visualization.run() 