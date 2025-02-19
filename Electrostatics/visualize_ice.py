import numpy as np
import pyvista as pv
from buchs_algorithm import create_ice_configuration

def visualize_ice_configuration(lattice_size=(3, 3, 3)):
    """
    Visualize the ice configuration generated by Buch's algorithm.
    
    Args:
        lattice_size: Tuple of (nx, ny, nz) defining the size of the ice lattice
    """
    # Generate ice configuration
    ice_config = create_ice_configuration(lattice_size)
    
    # Create PyVista plotter
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # Visualization parameters
    oxygen_radius = 0.3
    hydrogen_radius = 0.15
    bond_radius = 0.05
    
    # Add oxygen atoms (red)
    for molecule in ice_config:
        # Add oxygen sphere
        sphere_o = pv.Sphere(radius=oxygen_radius, center=molecule.oxygen_pos)
        plotter.add_mesh(sphere_o, color='red', opacity=0.8)
        
        # Add hydrogen atoms (white) and O-H bonds
        for h_pos in molecule.hydrogen_pos:
            # Add hydrogen sphere
            sphere_h = pv.Sphere(radius=hydrogen_radius, center=h_pos)
            plotter.add_mesh(sphere_h, color='white')
            
            # Add O-H bond
            bond_points = np.vstack((molecule.oxygen_pos, h_pos))
            bond = pv.Line(bond_points[0], bond_points[1])
            plotter.add_mesh(bond, color='black', line_width=3)
    
    # Add title
    plotter.add_text(
        f"Ice Configuration ({lattice_size[0]}x{lattice_size[1]}x{lattice_size[2]})",
        font_size=12
    )
    
    # Set camera position for better view
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.5)
    
    # Show the visualization
    plotter.show()

if __name__ == "__main__":
    # Create and visualize a 3x3x3 ice lattice
    visualize_ice_configuration((3, 3, 3)) 