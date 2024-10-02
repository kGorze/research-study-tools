import numpy as np
from scipy.ndimage import label

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button, Select, HoverTool
from bokeh.layouts import column, row
from bokeh.palettes import Category20

# Grid size
GRID_SIZE = 50

# Function to generate the lattice based on probability p and percolation model
def generate_lattice(p, model='site'):
    if model == 'site':
        # Site percolation: sites are occupied with probability p
        lattice = np.random.rand(GRID_SIZE, GRID_SIZE) < p
    elif model == 'bond':
        # Bond percolation model can be implemented here
        # Placeholder for bond percolation implementation
        lattice = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)  # All sites occupied
    else:
        raise ValueError("Model must be 'site' or 'bond'")
    return lattice

# Function to identify clusters in the lattice
def identify_clusters(lattice):
    # Use 4-connectivity
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=bool)
    labeled_lattice, num_clusters = label(lattice, structure=structure)
    return labeled_lattice, num_clusters


# Function to detect percolating clusters
def detect_percolating_clusters(labeled_lattice):
    percolating_clusters = set()
    # Vertical percolation
    top_labels = set(labeled_lattice[0, :])
    bottom_labels = set(labeled_lattice[-1, :])
    vertical_perc = top_labels.intersection(bottom_labels)
    percolating_clusters.update(vertical_perc)
    # Horizontal percolation
    left_labels = set(labeled_lattice[:, 0])
    right_labels = set(labeled_lattice[:, -1])
    horizontal_perc = left_labels.intersection(right_labels)
    percolating_clusters.update(horizontal_perc)
    # Remove background label (0)
    percolating_clusters.discard(0)
    return percolating_clusters

# Function to create the data source for plotting
def create_data_source(lattice, labeled_lattice, percolating_clusters):
    x_coords, y_coords = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    lattice_flat = lattice.flatten()
    labels_flat = labeled_lattice.flatten()
    colors = []
    
    # Generate color palette
    unique_labels = np.unique(labels_flat)
    unique_labels = unique_labels[unique_labels != 0]
    num_clusters = len(unique_labels)
    palette = Category20[20] * (num_clusters // 20 + 1)
    label_to_color = {}
    for idx, label in enumerate(unique_labels):
        if label in percolating_clusters:
            label_to_color[label] = 'red'  # Percolating clusters in red
        else:
            label_to_color[label] = palette[idx % len(palette)]
    # Calculate cluster sizes
    labels, counts = np.unique(labels_flat[labels_flat != 0], return_counts=True)
    label_to_size = dict(zip(labels, counts))
    cluster_size_list = []
    percolating_list = []
    for label in labels_flat:
        if label == 0:
            colors.append('white')
            cluster_size_list.append(0)
            percolating_list.append(False)
        else:
            colors.append(label_to_color[label])
            cluster_size_list.append(label_to_size[label])
            percolating_list.append(label in percolating_clusters)
    source = ColumnDataSource(data=dict(
        x=x_flat,
        y=y_flat,
        label=labels_flat,
        color=colors,
        occupied=lattice_flat,
        cluster_size=cluster_size_list,
        percolating=percolating_list
    ))
    return source, label_to_size

# Initial lattice generation
p_value = 0.5
lattice = generate_lattice(p_value, model='site')
labeled_lattice, num_clusters = identify_clusters(lattice)
percolating_clusters = detect_percolating_clusters(labeled_lattice)
source, cluster_sizes = create_data_source(lattice, labeled_lattice, percolating_clusters)

# Plot setup
plot = figure(width=600, height=600, title="Percolation Lattice",
              tools="pan,wheel_zoom,reset,hover,tap", match_aspect=True)

plot.grid.visible = False
plot.axis.visible = False
plot.rect(x='x', y='y', width=1, height=1, source=source, fill_color='color', line_color='black')

# Hover tool configuration
hover = plot.select_one(HoverTool)
hover.tooltips = [
    ("Cluster Label", "@label"),
    ("Cluster Size", "@cluster_size"),
    ("Percolating", "@percolating")
]

# Histogram for cluster size distribution
histogram_fig = figure(width=400, height=300, title="Cluster Size Distribution")
hist_source = ColumnDataSource(data=dict(cluster_size=[], counts=[]))
histogram_fig.vbar(x='cluster_size', top='counts', width=1, source=hist_source)

# Function to update the histogram
def update_histogram():
    cluster_sizes = source.data['cluster_size']
    sizes = [size for size in cluster_sizes if size > 0]
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    hist_source.data = dict(cluster_size=unique_sizes, counts=counts)

# Function to update the lattice when sliders change
def update_lattice(attr, old, new):
    p_value = p_slider.value
    model_value = percolation_model_select.value
    lattice = generate_lattice(p_value, model=model_value)
    labeled_lattice, num_clusters = identify_clusters(lattice)
    percolating_clusters = detect_percolating_clusters(labeled_lattice)
    new_source, cluster_sizes = create_data_source(lattice, labeled_lattice, percolating_clusters)
    source.data.update(new_source.data)
    update_histogram()

# Widgets
p_slider = Slider(title="Occupation Probability (p)", value=p_value, start=0.0, end=1.0, step=0.01)
percolation_model_select = Select(title="Percolation Model", value="site", options=["site", "bond"])
animate_button = Button(label="Play", width=60)
reset_button = Button(label="Reset", width=60)
randomize_button = Button(label="Randomize", width=80)

# Callbacks
p_slider.on_change('value', update_lattice)
percolation_model_select.on_change('value', update_lattice)

is_animating = False
def animate():
    p_value = p_slider.value + 0.01
    if p_value > 1.0:
        p_value = 0.0
    p_slider.value = p_value

def animate_update():
    if is_animating:
        animate()

def animate_button_clicked():
    global is_animating
    if animate_button.label == "Play":
        animate_button.label = "Pause"
        is_animating = True
    else:
        animate_button.label = "Play"
        is_animating = False

def reset_button_clicked():
    p_slider.value = 0.5
    update_lattice(None, None, None)

def randomize_button_clicked():
    p_value = np.random.rand()
    p_slider.value = p_value
    update_lattice(None, None, None)

animate_button.on_click(animate_button_clicked)
reset_button.on_click(reset_button_clicked)
randomize_button.on_click(randomize_button_clicked)
curdoc().add_periodic_callback(animate_update, 100)

# Initial histogram update
update_histogram()

# Layout
controls = row(p_slider, percolation_model_select, animate_button, reset_button, randomize_button)
layout = column(controls, row(plot, histogram_fig))
curdoc().add_root(layout)
curdoc().title = "Percolation Simulation"
