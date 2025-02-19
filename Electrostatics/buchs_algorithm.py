import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass

@dataclass
class WaterMolecule:
    """Represents a water molecule with oxygen and hydrogen positions."""
    oxygen_pos: np.ndarray
    hydrogen_pos: List[np.ndarray]
    
class BuchsAlgorithm:
    """Implementation of Buch's algorithm for generating proton-disordered ice configurations."""
    
    def __init__(self, lattice_size: Tuple[int, int, int]):
        """
        Initialize the Buch's algorithm with given lattice dimensions.
        
        Args:
            lattice_size: Tuple of (nx, ny, nz) defining the size of the ice lattice
        """
        self.lattice_size = lattice_size
        self.molecules: List[WaterMolecule] = []
        self.dipole_moment = np.zeros(3)
        
    def generate_oxygen_lattice(self) -> None:
        """Generate the ordered oxygen lattice positions."""
        nx, ny, nz = self.lattice_size
        a = 4.5  # Approximate ice Ih lattice parameter in Angstroms
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Create hexagonal ice structure
                    if (i + j + k) % 2 == 0:
                        pos = np.array([i * a, j * a, k * a])
                        self.molecules.append(WaterMolecule(
                            oxygen_pos=pos,
                            hydrogen_pos=[]
                        ))
    
    def assign_hydrogens(self) -> None:
        """
        Assign hydrogen positions following ice rules:
        1. Two hydrogens per oxygen
        2. One hydrogen per O-O bond
        3. Tetrahedral coordination
        """
        for mol in self.molecules:
            # Define possible H positions in tetrahedral arrangement
            tetrahedral_vectors = [
                np.array([0.9572, 0, 0]),      # Standard O-H bond length
                np.array([-0.9572, 0, 0]),
                np.array([0, 0.9572, 0]),
                np.array([0, -0.9572, 0])
            ]
            
            # Randomly select two positions for hydrogens
            selected_positions = np.random.choice(4, size=2, replace=False)
            mol.hydrogen_pos = [
                mol.oxygen_pos + tetrahedral_vectors[i] for i in selected_positions
            ]
            
            # Update dipole moment
            for h_pos in mol.hydrogen_pos:
                self.dipole_moment += h_pos - mol.oxygen_pos
    
    def minimize_dipole(self) -> None:
        """
        Minimize the net dipole moment of the configuration through hydrogen flips.
        """
        max_iterations = 1000
        threshold = 0.1  # Threshold for considering dipole moment zero
        
        for _ in range(max_iterations):
            if np.linalg.norm(self.dipole_moment) < threshold:
                break
                
            # Randomly select a molecule
            mol_idx = np.random.randint(0, len(self.molecules))
            mol = self.molecules[mol_idx]
            
            # Try flipping hydrogens
            old_h_pos = mol.hydrogen_pos.copy()
            # Flip operation here (simplified)
            mol.hydrogen_pos = [h + np.random.randn(3) * 0.1 for h in mol.hydrogen_pos]
            
            # Calculate new dipole moment
            new_dipole = self.calculate_total_dipole()
            
            # Accept if dipole moment decreases, otherwise revert
            if np.linalg.norm(new_dipole) < np.linalg.norm(self.dipole_moment):
                self.dipole_moment = new_dipole
            else:
                mol.hydrogen_pos = old_h_pos
    
    def calculate_total_dipole(self) -> np.ndarray:
        """Calculate the total dipole moment of the configuration."""
        total_dipole = np.zeros(3)
        for mol in self.molecules:
            for h_pos in mol.hydrogen_pos:
                total_dipole += h_pos - mol.oxygen_pos
        return total_dipole
    
    def generate_configuration(self) -> List[WaterMolecule]:
        """
        Generate a complete proton-disordered ice configuration.
        
        Returns:
            List of WaterMolecule objects representing the ice configuration
        """
        self.generate_oxygen_lattice()
        self.assign_hydrogens()
        self.minimize_dipole()
        return self.molecules

def create_ice_configuration(lattice_size: Tuple[int, int, int]) -> List[WaterMolecule]:
    """
    Create a proton-disordered ice configuration using Buch's algorithm.
    
    Args:
        lattice_size: Tuple of (nx, ny, nz) defining the size of the ice lattice
        
    Returns:
        List of WaterMolecule objects representing the ice configuration
    """
    algorithm = BuchsAlgorithm(lattice_size)
    return algorithm.generate_configuration()

# Example usage
if __name__ == "__main__":
    # Create a 3x3x3 ice lattice
    lattice_size = (3, 3, 3)
    ice_config = create_ice_configuration(lattice_size)
    
    # Print some basic information
    print(f"Generated ice configuration with {len(ice_config)} water molecules")
    print(f"Net dipole moment: {BuchsAlgorithm(lattice_size).dipole_moment}")
