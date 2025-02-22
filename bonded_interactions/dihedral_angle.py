import matplotlib
# Use TkAgg backend
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

class DihedralAngleSimulation:
    def __init__(self):
        # Simulation parameters
        self.V_n = 2.5    # Reduced barrier height for smoother motion
        self.n = 3        # Periodicity (e.g., 3 for sp3 carbons)
        self.delta = 0.0  # Phase offset
        self.dt = 0.02    # Time step
        self.moment_of_inertia = 1.0  # Moment of inertia for rotation
        
        # Initial conditions
        self.phi = np.pi/2  # Initial dihedral angle (90 degrees)
        self.omega = 0.5    # Initial angular velocity for continuous rotation
        
        # Fixed parameters
        self.bond_length = 1.5  # Fixed bond length
        self.bond_angle = 109.5 * np.pi/180  # Tetrahedral angle (sp3)
        
        # Calculate maximum energy
        self.max_energy = self.V_n  # Maximum energy is V_n
        
    def calculate_torque(self, phi):
        """Calculate the torque from the periodic potential"""
        return 0.5 * self.V_n * self.n * np.sin(self.n * phi - self.delta)
    
    def calculate_energy(self, phi):
        """Calculate the potential energy using the periodic function"""
        return 0.5 * self.V_n * (1 + np.cos(self.n * phi - self.delta))
    
    def update(self):
        """Update dihedral angle and angular velocity using Verlet integration"""
        torque = self.calculate_torque(self.phi)
        angular_acceleration = torque / self.moment_of_inertia
        
        # Update angle and angular velocity using Verlet algorithm
        phi_new = self.phi + self.omega * self.dt + 0.5 * angular_acceleration * self.dt**2
        torque_new = self.calculate_torque(phi_new)
        angular_acceleration_new = torque_new / self.moment_of_inertia
        omega_new = self.omega + 0.5 * (angular_acceleration + angular_acceleration_new) * self.dt
        
        self.phi = phi_new
        self.omega = omega_new
        
        # Normalize angle to [0, 2π]
        self.phi = self.phi % (2 * np.pi)
        
        energy = self.calculate_energy(self.phi)
        return self.phi, energy

class DihedralAngleVisualization:
    def __init__(self):
        self.simulation = DihedralAngleSimulation()
        
        # Setup the figure and animation
        plt.style.use('default')  # Cleaner style for 3D visualization
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.patch.set_facecolor('white')
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
        
        # 3D atoms visualization subplot
        self.ax1 = self.fig.add_subplot(gs[0], projection='3d')
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-3, 3)
        self.ax1.set_zlim(-3, 3)
        self.ax1.set_title('Dihedral Angle Rotation', pad=20, fontsize=12, fontweight='bold')
        
        # Make the 3D plot clearer
        self.ax1.xaxis.pane.fill = False
        self.ax1.yaxis.pane.fill = False
        self.ax1.zaxis.pane.fill = False
        self.ax1.grid(True, alpha=0.3)
        
        # Energy plot subplot
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(0, self.simulation.max_energy * 1.1)
        self.ax2.set_xlabel('Time (arbitrary units)', fontsize=10)
        self.ax2.set_ylabel('Potential Energy', fontsize=10)
        self.ax2.set_title('Energy vs Time', pad=20, fontsize=12, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('#f8f9fa')
        
        # Initialize atom positions
        self.atom_positions = self.calculate_atom_positions(self.simulation.phi)
        
        # Define colors and labels for atoms
        self.atom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        self.atom_labels = ['C1', 'C2', 'C3', 'C4']
        
        # Initialize visualization elements
        # Create scatter plots for each atom separately for proper labeling
        self.atoms = []
        for i in range(4):
            atom = self.ax1.scatter([], [], [], c=self.atom_colors[i], s=200, label=self.atom_labels[i])
            self.atoms.append(atom)
        
        self.bonds, = self.ax1.plot([], [], [], 'k-', linewidth=2)
        self.energy_line, = self.ax2.plot([], [], '#2ca02c', linewidth=2, label='Potential Energy')
        
        # Add reference plane with transparency
        self.reference_plane = None
        self.update_reference_plane(self.simulation.phi)
        
        # Add legends
        self.ax1.legend(loc='upper left', framealpha=0.9)
        self.ax2.legend(loc='upper right', framealpha=0.9)
        
        # Text displays for current values
        self.text_display = self.ax2.text(0.02, 0.98, '', transform=self.ax2.transAxes,
                                        verticalalignment='top', fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='white',
                                                 edgecolor='gray', alpha=0.9))
        
        # Add axis labels
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        
        # Data storage
        self.t_data = []
        self.e_data = []
        self.max_energy_seen = 0
        
        # Adjust layout
        plt.tight_layout()
    
    def calculate_atom_positions(self, phi):
        """Calculate positions of all four atoms given the dihedral angle"""
        L = self.simulation.bond_length
        theta = self.simulation.bond_angle
        
        # First three atoms define the reference plane
        p1 = np.array([0, 0, 0])
        p2 = np.array([L, 0, 0])
        p3 = np.array([L + L*np.cos(theta), L*np.sin(theta), 0])
        
        # Fourth atom position depends on the dihedral angle
        r = L * np.sin(theta)
        h = L * np.cos(theta)
        p4_x = p3[0] + h*np.cos(phi)
        p4_y = p3[1] + r*np.cos(phi)
        p4_z = r*np.sin(phi)
        p4 = np.array([p4_x, p4_y, p4_z])
        
        return np.vstack([p1, p2, p3, p4])
    
    def update_reference_plane(self, phi):
        """Update the reference plane visualization"""
        positions = self.calculate_atom_positions(phi)
        x = positions[:3, 0]
        y = positions[:3, 1]
        z = positions[:3, 2]
        
        # Clear previous plane if it exists
        if self.reference_plane is not None:
            self.reference_plane.remove()
        
        # Plot new reference plane
        self.reference_plane = self.ax1.plot_trisurf(x, y, z, alpha=0.1, color='gray')
    
    def init(self):
        """Initialize animation"""
        for atom in self.atoms:
            atom._offsets3d = ([], [], [])
        self.bonds.set_data([], [])
        self.bonds.set_3d_properties([])
        self.energy_line.set_data([], [])
        return tuple(self.atoms) + (self.bonds, self.energy_line)
    
    def animate(self, frame):
        """Animation update function"""
        # Update simulation
        phi, energy = self.simulation.update()
        
        # Track maximum energy
        self.max_energy_seen = max(self.max_energy_seen, energy)
        
        # Update atom positions
        positions = self.calculate_atom_positions(phi)
        
        # Update each atom separately
        for i, atom in enumerate(self.atoms):
            atom._offsets3d = ([positions[i,0]], [positions[i,1]], [positions[i,2]])
        
        # Update bonds
        self.bonds.set_data(positions[:, 0], positions[:, 1])
        self.bonds.set_3d_properties(positions[:, 2])
        
        # Update energy plot
        self.t_data.append(frame * self.simulation.dt)
        self.e_data.append(energy)
        self.energy_line.set_data(self.t_data, self.e_data)
        
        # Update text display with more information
        status_text = (f'Dihedral Angle: {np.degrees(phi):.1f}°\n'
                      f'Periodicity: {self.simulation.n}\n'
                      f'Energy: {energy:.2f}\n'
                      f'Max Energy: {self.max_energy_seen:.2f}\n'
                      f'Period: {360/self.simulation.n:.1f}°')
        self.text_display.set_text(status_text)
        
        # Rotate view for better 3D perspective
        self.ax1.view_init(elev=20, azim=frame/8)
        
        return tuple(self.atoms) + (self.bonds, self.energy_line, self.text_display)
    
    def run(self):
        """Run the animation"""
        anim = FuncAnimation(self.fig, self.animate, init_func=self.init,
                           frames=1000, interval=20, blit=True)
        plt.show()

if __name__ == "__main__":
    try:
        print("Starting visualization...")
        visualization = DihedralAngleVisualization()
        print("Visualization object created successfully")
        print("Running animation...")
        visualization.run()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Script finished executing") 