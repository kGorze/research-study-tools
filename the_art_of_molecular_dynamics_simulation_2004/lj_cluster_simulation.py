import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import dash

# Initialize the Dash app
app = Dash(__name__)
server = app.server

# Simulation parameters
N = 100  # Default number of particles
L = 10.0  # Box length
T = 1.0  # Temperature
dt = 0.01  # Time step
is_paused = False
model = 'Lennard-Jones'  # Default model
dimension = '3D'  # Default dimension

# Initialize particle positions and velocities
def initialize_particles():
    global positions, velocities
    positions = np.random.uniform(0, L, (N, 3))
    velocities = np.random.normal(0, np.sqrt(T), (N, 3))

initialize_particles()

# Global variables to store simulation state
energies = {}
labels = np.array([])

def lj_potential(r):
    """Compute Lennard-Jones potential."""
    return 4 * ((1 / r)**12 - (1 / r)**6)

def lj_force(r_vec):
    """Compute Lennard-Jones force."""
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)
    magnitude = 48 * ((1 / r)**13) - 24 * ((1 / r)**7)
    return magnitude * (r_vec / r)

def soft_sphere_potential(r):
    """Compute Soft-Sphere potential."""
    return (1 / r)**12

def soft_sphere_force(r_vec):
    """Compute Soft-Sphere force."""
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)
    magnitude = 12 * (1 / r)**13
    return magnitude * (r_vec / r)

def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            # Apply minimum image convention for periodic boundaries
            r_vec -= L * np.round(r_vec / L)
            if model == 'Lennard-Jones':
                force = lj_force(r_vec)
            else:
                force = soft_sphere_force(r_vec)
            forces[i] += force
            forces[j] -= force  # Newton's third law
    return forces

def simulate_step():
    global positions, velocities
    # Compute forces
    forces = compute_forces(positions)
    # Update positions
    positions += velocities * dt + 0.5 * forces * dt**2
    # Apply periodic boundary conditions
    positions %= L
    # Update velocities
    new_forces = compute_forces(positions)
    velocities += 0.5 * (forces + new_forces) * dt

def identify_clusters():
    global labels
    # Use DBSCAN clustering
    eps_value = 1.5 if dimension == '3D' else 1.5
    clustering = DBSCAN(eps=eps_value, min_samples=2).fit(positions[:, :2] if dimension == '2D' else positions)
    labels = clustering.labels_

def compute_binding_energy():
    global energies
    energies = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip noise points
        indices = np.where(labels == cluster_id)[0]
        energy = 0.0
        for i in indices:
            for j in indices:
                if i < j:
                    r_vec = positions[i] - positions[j]
                    r_vec -= L * np.round(r_vec / L)
                    r = np.linalg.norm(r_vec)
                    if r == 0:
                        continue
                    if model == 'Lennard-Jones':
                        energy += lj_potential(r)
                    else:
                        energy += soft_sphere_potential(r)
        energies[cluster_id] = energy

def update_cluster_info(cluster_id):
    num_particles = np.sum(labels == cluster_id)
    cluster_positions = positions[labels == cluster_id]
    avg_distance = np.mean(cdist(cluster_positions, cluster_positions))
    total_energy = energies[cluster_id]
    info = {
        'Cluster ID': cluster_id,
        'Number of Particles': num_particles,
        'Average Distance': avg_distance,
        'Total Binding Energy': total_energy
    }
    return info

def create_figure():
    df = positions.copy()
    labels_unique = np.unique(labels)
    data = []
    for cluster_id in labels_unique:
        idx = labels == cluster_id
        cluster_positions = df[idx]
        if cluster_id == -1:
            # Noise points
            cluster_color = 'rgba(200,200,200,0.5)'
            name = 'Noise'
        else:
            energy = energies[cluster_id]
            # Map energy to color
            norm_energy = (energy - min(energies.values())) / (max(energies.values()) - min(energies.values()) + 1e-10)
            cluster_color = f'hsl({int(240 * norm_energy)}, 100%, 50%)'
            name = f'Cluster {cluster_id}'
        if dimension == '3D':
            trace = go.Scatter3d(
                x=cluster_positions[:, 0],
                y=cluster_positions[:, 1],
                z=cluster_positions[:, 2],
                mode='markers',
                marker=dict(size=5, color=cluster_color),
                name=name,
                customdata=[cluster_id]*cluster_positions.shape[0],
                hovertemplate='Cluster ID: %{customdata}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
            )
        else:
            trace = go.Scatter(
                x=cluster_positions[:, 0],
                y=cluster_positions[:, 1],
                mode='markers',
                marker=dict(size=8, color=cluster_color),
                name=name,
                customdata=[cluster_id]*cluster_positions.shape[0],
                hovertemplate='Cluster ID: %{customdata}<br>(%{x:.2f}, %{y:.2f})<extra></extra>'
            )
        data.append(trace)
    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[0, L]),
            yaxis=dict(range=[0, L]),
            zaxis=dict(range=[0, L]),
        ) if dimension == '3D' else dict(
            xaxis=dict(range=[0, L]),
            yaxis=dict(range=[0, L]),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode='closest',
        showlegend=True
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

def create_histogram():
    sizes = [np.sum(labels == cid) for cid in energies.keys()]
    energy_values = list(energies.values())
    hist_sizes = go.Histogram(
        x=sizes,
        nbinsx=10,
        name='Cluster Sizes'
    )
    hist_energies = go.Histogram(
        x=energy_values,
        nbinsx=10,
        name='Binding Energies'
    )
    fig = go.Figure(data=[hist_sizes, hist_energies])
    fig.update_layout(
        barmode='overlay',
        title='Cluster Sizes and Binding Energies Distribution',
        xaxis_title='Value',
        yaxis_title='Frequency'
    )
    fig.update_traces(opacity=0.75)
    return fig

# Dash Layout
app.layout = html.Div([
    html.H1('Lennard-Jones Fluid Cluster Analysis'),
    html.Div([
        html.Div([
            html.Label('Number of Particles'),
            dcc.Slider(
                id='num-particles-slider',
                min=10,
                max=500,
                step=10,
                value=N,
                marks={i: str(i) for i in range(10, 501, 50)}
            ),
            html.Br(),
            html.Label('Temperature'),
            dcc.Slider(
                id='temperature-slider',
                min=0.1,
                max=5.0,
                step=0.1,
                value=T,
                marks={i: str(i) for i in np.arange(0.1, 5.1, 0.5)}
            ),
            html.Br(),
            html.Label('Particle Density'),
            dcc.Slider(
                id='density-slider',
                min=0.1,
                max=1.0,
                step=0.1,
                value=N / L**3,
                marks={i/10: f'{i/10:.1f}' for i in range(1, 11)}
            ),
            html.Br(),
            html.Label('Model'),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Lennard-Jones', 'value': 'Lennard-Jones'},
                    {'label': 'Soft-Sphere', 'value': 'Soft-Sphere'}
                ],
                value=model
            ),
            html.Br(),
            html.Label('Dimension'),
            dcc.RadioItems(
                id='dimension-radio',
                options=[
                    {'label': '2D', 'value': '2D'},
                    {'label': '3D', 'value': '3D'}
                ],
                value=dimension
            ),
            html.Br(),
            html.Button('Reset', id='reset-button', n_clicks=0),
            html.Button('Randomize Positions', id='randomize-button', n_clicks=0),
            html.Button('Pause', id='pause-play-button', n_clicks=0),
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        html.Div([
            dcc.Graph(id='particle-graph', style={'height': '600px'}),
            dcc.Graph(id='histogram-graph', style={'height': '300px'}),
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'}),
    ]),
    html.Div(id='cluster-info', style={'padding': '20px'}),
    dcc.Interval(
        id='interval-component',
        interval=500,  # Update every 500 milliseconds
        n_intervals=0
    ),
    # Hidden div to store data
    html.Div(id='simulation-data', style={'display': 'none'})
])

@app.callback(
    Output('simulation-data', 'children'),
    Input('interval-component', 'n_intervals'),
    State('simulation-data', 'children'),
    State('pause-play-button', 'n_clicks'),
    State('num-particles-slider', 'value'),
    State('temperature-slider', 'value'),
    State('density-slider', 'value'),
    State('model-dropdown', 'value'),
    State('dimension-radio', 'value'),
    prevent_initial_call=True
)
def update_simulation(n_intervals, sim_data, pause_clicks, num_particles, temperature, density, selected_model, selected_dimension):
    global positions, velocities, labels, energies, model, is_paused, N, T, L, dimension

    # Handle pause/play
    ctx = dash.callback_context
    if ctx.triggered and 'pause-play-button' in ctx.triggered[0]['prop_id']:
        is_paused = not is_paused
        return sim_data

    if is_paused:
        return sim_data

    # Update parameters if they have changed
    changed = False
    if N != num_particles or T != temperature or model != selected_model or dimension != selected_dimension:
        N = num_particles
        T = temperature
        model = selected_model
        dimension = selected_dimension
        L = (N / density)**(1/3)
        initialize_particles()
        changed = True

    # Run simulation step
    simulate_step()

    # Cluster analysis
    identify_clusters()
    compute_binding_energy()

    # Store simulation data in hidden div
    sim_data = {
        'positions': positions.tolist(),
        'velocities': velocities.tolist(),
        'labels': labels.tolist(),
        'energies': energies,
        'dimension': dimension
    }

    return sim_data

@app.callback(
    Output('particle-graph', 'figure'),
    Input('simulation-data', 'children')
)
def update_graph(sim_data):
    if sim_data is None:
        return dash.no_update
    global positions, labels, energies, dimension
    positions = np.array(sim_data['positions'])
    labels = np.array(sim_data['labels'])
    energies = sim_data['energies']
    dimension = sim_data['dimension']
    fig = create_figure()
    return fig

@app.callback(
    Output('histogram-graph', 'figure'),
    Input('simulation-data', 'children')
)
def update_histogram_graph(sim_data):
    if sim_data is None:
        return dash.no_update
    fig = create_histogram()
    return fig

@app.callback(
    Output('cluster-info', 'children'),
    Input('particle-graph', 'clickData')
)
def display_cluster_info(clickData):
    if clickData is None:
        return 'Click on a cluster to see details.'
    cluster_id = clickData['points'][0]['customdata']
    if cluster_id == -1:
        return 'Noise point clicked.'
    info = update_cluster_info(cluster_id)
    info_html = html.Div([
        html.H3(f'Cluster {cluster_id} Details'),
        html.P(f"Number of Particles: {info['Number of Particles']}"),
        html.P(f"Average Distance: {info['Average Distance']:.2f}"),
        html.P(f"Total Binding Energy: {info['Total Binding Energy']:.2f}")
    ])
    return info_html

@app.callback(
    Output('pause-play-button', 'children'),
    Input('pause-play-button', 'n_clicks')
)
def update_pause_play_button(n_clicks):
    if n_clicks % 2 == 0:
        return 'Pause'
    else:
        return 'Play'

@app.callback(
    Output('simulation-data', 'children'),
    Input('reset-button', 'n_clicks'),
    Input('randomize-button', 'n_clicks'),
    State('num-particles-slider', 'value'),
    State('temperature-slider', 'value'),
    State('density-slider', 'value'),
    State('model-dropdown', 'value'),
    State('dimension-radio', 'value'),
    prevent_initial_call=True
)
def reset_or_randomize_simulation(reset_clicks, randomize_clicks, num_particles, temperature, density, selected_model, selected_dimension):
    global positions, velocities, labels, energies, model, is_paused, N, T, L, dimension

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    N = num_particles
    T = temperature
    model = selected_model
    dimension = selected_dimension
    L = (N / density)**(1/3)

    # Initialize positions and velocities
    initialize_particles()
    is_paused = False

    # Cluster analysis
    identify_clusters()
    compute_binding_energy()

    sim_data = {
        'positions': positions.tolist(),
        'velocities': velocities.tolist(),
        'labels': labels.tolist(),
        'energies': energies,
        'dimension': dimension
    }

    return sim_data

if __name__ == '__main__':
    app.run_server(debug=True)
