## Exercise 4.7: Voronoi Analysis for Two-Dimensional Systems

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon

# Positions in 2D
positions = np.random.uniform(0, box_length, (num_particles, 2))

# Create Voronoi diagram
vor = Voronoi(positions)

# Plotting the Voronoi diagram
fig = voronoi_plot_2d(vor)
plt.title('Voronoi Diagram of 2D Particle System')
plt.show()

# Analyze cell properties
cell_areas = []
num_sides = []

for region_index in vor.point_region:
    vertices = vor.regions[region_index]
    if -1 in vertices or len(vertices) == 0:
        continue  # Skip infinite regions
    polygon = [vor.vertices[i] for i in vertices]
    poly = Polygon(polygon)
    area = poly.area
    cell_areas.append(area)
    num_sides.append(len(vertices))

# Plot histogram of number of sides
print(num_sides)
plt.hist(num_sides, bins=np.arange(2.5, 10.5, 1), edgecolor='black')
plt.xlabel('Number of Sides')
plt.ylabel('Frequency')
plt.title('Distribution of Voronoi Cell Sides')
plt.show()
