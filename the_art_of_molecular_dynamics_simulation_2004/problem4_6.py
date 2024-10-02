## Exercise 4.6: Structural Analysis with Three-Atom Correlations
import numpy as np
import matplotlib.pyplot as plt

# Function to compute angle distribution
def compute_angle_distribution(positions, neighbor_list, box_length):
    angles = []
    for i in range(len(positions)):
        neighbors = neighbor_list[i]
        for j in range(len(neighbors)):
            for k in range(j+1, len(neighbors)):
                idx_j = neighbors[j]
                idx_k = neighbors[k]
                vec_ij = positions[idx_j] - positions[i]
                vec_ik = positions[idx_k] - positions[i]
                
                # Apply periodic boundary conditions
                vec_ij = vec_ij - box_length * np.round(vec_ij / box_length)
                vec_ik = vec_ik - box_length * np.round(vec_ik / box_length)
                
                # Compute the angle
                dot_product = np.dot(vec_ij, vec_ik)
                norm_ij = np.linalg.norm(vec_ij)
                norm_ik = np.linalg.norm(vec_ik)
                
                # Ensure we don't divide by zero
                if norm_ij > 0 and norm_ik > 0:
                    cos_theta = dot_product / (norm_ij * norm_ik)
                    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                    angles.append(np.degrees(angle))
    return angles

# Example usage
box_length = 10.0  # Define box length (example value)
# Assuming positions and neighbor_list are defined elsewhere in the script

# Compute angle distribution
angles = compute_angle_distribution(positions, neighbor_list, box_length)

# Plotting
plt.hist(angles, bins=180, range=(0, 180))
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.title('Distribution of Bond Angles')
plt.show()
