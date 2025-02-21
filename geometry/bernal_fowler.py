import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Arc

def create_water_molecule(center=(0,0,0), orientation=None):
    """
    Create a single water molecule with Bernal-Fowler/TIP4P geometry.
    
    Parameters:
        center: (x,y,z) coordinates of the oxygen atom
        orientation: Optional rotation matrix to orient the molecule
        
    Returns:
        Dictionary containing coordinates of all sites
    """
    # Convert parameters to Angstroms and degrees
    OH_DISTANCE = 0.9572  # Å
    HOH_ANGLE = 104.52   # degrees
    OM_DISTANCE = 0.15   # Å (typical for TIP4P)
    
    # Convert angle to radians
    theta = np.radians(HOH_ANGLE/2)
    
    # Calculate positions relative to oxygen at origin
    h1_pos = np.array([OH_DISTANCE * np.cos(theta), OH_DISTANCE * np.sin(theta), 0])
    h2_pos = np.array([OH_DISTANCE * np.cos(theta), -OH_DISTANCE * np.sin(theta), 0])
    m_pos = np.array([OM_DISTANCE, 0, 0])  # M-site along bisector
    
    # Apply orientation if provided
    if orientation is not None:
        h1_pos = orientation @ h1_pos
        h2_pos = orientation @ h2_pos
        m_pos = orientation @ m_pos
    
    # Shift to center position
    center = np.array(center)
    return {
        'O': center,
        'H1': center + h1_pos,
        'H2': center + h2_pos,
        'M': center + m_pos
    }

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix for rotation around axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def create_tetrahedral_cluster():
    """
    Create a cluster of 5 water molecules in tetrahedral coordination.
    Central molecule with 4 neighbors arranged tetrahedrally.
    """
    # Distance between oxygens in ice Ih
    OO_DISTANCE = 2.76  # Å
    
    # Create central molecule rotated for better view
    rot = rotation_matrix([0, 1, 0], np.pi/6)
    central = create_water_molecule((0,0,0), rot)
    
    # Create 4 neighboring molecules at tetrahedral positions
    neighbors = []
    
    # Tetrahedral vertices with proper orientation for hydrogen bonding
    vertices = [
        ((OO_DISTANCE, 0, 0), rotation_matrix([0, 1, 0], np.pi/3)),
        ((-OO_DISTANCE/3, OO_DISTANCE*np.cos(np.pi/6), OO_DISTANCE*np.sin(np.pi/6)), 
         rotation_matrix([1, 0, 1], 2*np.pi/3)),
        ((-OO_DISTANCE/3, -OO_DISTANCE*np.cos(np.pi/6), OO_DISTANCE*np.sin(np.pi/6)),
         rotation_matrix([1, 0, -1], 2*np.pi/3)),
        ((-OO_DISTANCE/3, 0, -OO_DISTANCE*np.sqrt(8/9)),
         rotation_matrix([1, 1, 0], 2*np.pi/3))
    ]
    
    for vertex, orientation in vertices:
        neighbors.append(create_water_molecule(vertex, orientation))
    
    return central, neighbors

def plot_water_molecule(ax, molecule, color='b', show_m_site=True, show_angle=False, show_labels=True):
    """Plot a single water molecule with all interaction sites."""
    # Plot O-H bonds with thicker lines
    for h in ['H1', 'H2']:
        ax.plot([molecule['O'][0], molecule[h][0]],
                [molecule['O'][1], molecule[h][1]],
                [molecule['O'][2], molecule[h][2]], 'k-', linewidth=2.5)
        
        # Add OH distance labels if requested
        if show_labels:
            mid_point = (molecule['O'] + molecule[h]) / 2
            ax.text(mid_point[0], mid_point[1], mid_point[2], 
                   '0.9572 Å', fontsize=8, color='black',
                   bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Plot atoms with larger sizes and better colors
    ax.scatter(*molecule['O'], color='red', s=200, label='O (LJ center)')
    ax.scatter(*molecule['H1'], color='white', edgecolor='black', s=100, label='H (+qH)')
    ax.scatter(*molecule['H2'], color='white', edgecolor='black', s=100)
    
    if show_m_site:
        # Make M-site more visible
        ax.scatter(*molecule['M'], color='blue', s=80, alpha=0.9, label='M (-2qH)')
        # Add line from O to M-site with better visibility
        ax.plot([molecule['O'][0], molecule['M'][0]],
                [molecule['O'][1], molecule['M'][1]],
                [molecule['O'][2], molecule['M'][2]], 'b-', linewidth=1.5, alpha=0.7)
        
        # Add M-site distance annotation with background
        mid_point = (molecule['O'] + molecule['M']) / 2
        ax.text(mid_point[0], mid_point[1], mid_point[2], 
                '0.15 Å', fontsize=8, color='blue',
                bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    if show_angle:
        # Calculate vectors for angle visualization
        v1 = molecule['H1'] - molecule['O']
        v2 = molecule['H2'] - molecule['O']
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # Add angle arc with better visibility
        radius = 0.4
        theta = np.linspace(0, angle, 50)
        x = molecule['O'][0] + radius * np.cos(theta - angle/2)
        y = molecule['O'][1] + radius * np.sin(theta - angle/2)
        z = np.full_like(x, molecule['O'][2])
        ax.plot(x, y, z, 'k:', linewidth=1.5, alpha=0.7)
        # Add angle label
        arc_mid = np.array([radius * np.cos(angle/2), radius * np.sin(angle/2), 0])
        ax.text(molecule['O'][0] + arc_mid[0], molecule['O'][1] + arc_mid[1], molecule['O'][2],
                '104.52°', fontsize=8, bbox=dict(facecolor='white', alpha=0.7, pad=1))

def plot_bernal_fowler_geometry():
    """Create plots showing Bernal-Fowler water geometry."""
    fig = plt.figure(figsize=(18, 6))  # Wider figure for better spacing
    
    # Single molecule view with interaction sites
    ax1 = fig.add_subplot(131, projection='3d')
    rot = rotation_matrix([0, 1, 0], np.pi/6)  # Rotate for better view
    molecule = create_water_molecule(orientation=rot)
    plot_water_molecule(ax1, molecule, show_angle=True)
    ax1.set_title('TIP4P/Bernal-Fowler Model\nInteraction Sites', pad=20)
    
    # Set equal aspect ratio and proper limits for single molecule
    ax1.set_box_aspect([2,2,2])  # Force equal aspect ratio
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_zlim(-1.2, 1.2)
    ax1.view_init(elev=15, azim=45)  # Better angle for viewing
    
    # Add labels and annotations with interaction details
    ax1.text2D(0.02, 0.98, 
               'Interaction sites:\n'
               '• O: Lennard-Jones (ε,σ)\n'
               '• H: Charge (+qH)\n'
               '• M: Charge (-2qH)', 
               transform=ax1.transAxes,
               bbox=dict(facecolor='white', alpha=0.9, pad=5))
    
    # Add legend with better positioning
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 0.9))
    
    # Tetrahedral cluster view
    ax2 = fig.add_subplot(132, projection='3d')
    central, neighbors = create_tetrahedral_cluster()
    
    # Plot central molecule and neighbors
    plot_water_molecule(ax2, central, show_m_site=False, show_labels=False)
    for neighbor in neighbors:
        plot_water_molecule(ax2, neighbor, show_m_site=False, show_labels=False)
    ax2.set_title('Tetrahedral Coordination\n(5 Molecules)', pad=20)
    
    # Add O-O hydrogen bonds with better visibility
    for i, neighbor in enumerate(neighbors):
        # Plot hydrogen bond
        ax2.plot([central['O'][0], neighbor['O'][0]],
                 [central['O'][1], neighbor['O'][1]],
                 [central['O'][2], neighbor['O'][2]], 'b--', linewidth=2, alpha=0.7)
        
        # Add O-O distance annotation for first neighbor
        if i == 0:
            mid_point = (central['O'] + neighbor['O']) / 2
            ax2.text(mid_point[0], mid_point[1], mid_point[2], 
                    'O-O: 2.76 Å', fontsize=10, bbox=dict(facecolor='white', alpha=0.9, pad=2))
    
    # Add tetrahedral angle visualization
    ax2.text2D(0.05, 0.95, 'Tetrahedral angle: 109.47°',
               transform=ax2.transAxes,
               bbox=dict(facecolor='white', alpha=0.9, pad=2))
    
    # Set proper view angle and limits with equal aspect ratio
    ax2.view_init(elev=25, azim=45)
    ax2.set_box_aspect([2,2,2])
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(-3, 3)
    
    # Tetrahedral arrangement view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.view_init(elev=30, azim=60)  # Better angle for tetrahedral view
    plot_water_molecule(ax3, central, show_m_site=False, show_labels=False)
    
    # Plot neighbors with enhanced hydrogen bonds
    for neighbor in neighbors:
        plot_water_molecule(ax3, neighbor, show_m_site=False, show_labels=False)
        ax3.plot([central['O'][0], neighbor['O'][0]],
                 [central['O'][1], neighbor['O'][1]],
                 [central['O'][2], neighbor['O'][2]], 'b--', linewidth=2, alpha=0.7)
    
    ax3.set_title('Tetrahedral Arrangement\n(Hydrogen Bond Network)', pad=20)
    
    # Set proper limits and view with equal aspect ratio
    ax3.set_box_aspect([2,2,2])
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.set_zlim(-3, 3)
    
    # Adjust all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X (Å)', labelpad=10)
        ax.set_ylabel('Y (Å)', labelpad=10)
        ax.set_zlabel('Z (Å)', labelpad=10)
        # Make grid lighter but more visible
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Make panes slightly visible
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
    
    plt.tight_layout(w_pad=3)  # Increase spacing between subplots
    plt.savefig('bernal_fowler_geometry.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_bernal_fowler_geometry()
