import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from .example_gibbs_duhem import (
    lennard_jones_potential,
    calculate_total_energy,
    calculate_phase_x_g,
    get_realistic_phase_properties
)

# Set default plot style
plt.rcParams['figure.figsize'] = [8, 6]  # Smaller default size
plt.rcParams['font.size'] = 10  # Smaller font
plt.rcParams['lines.linewidth'] = 1.5  # Thinner lines
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 100  # Lower default DPI

def plot_lennard_jones_potential():
    """Plot the Lennard-Jones potential for different epsilon values."""
    r = np.linspace(0.8, 3.0, 1000)
    epsilon_ref = 1.0
    epsilon_new = 1.2
    
    # Calculate potentials
    U_ref = [lennard_jones_potential(np.array([ri]), 0, 0, epsilon=epsilon_ref) for ri in r]
    U_new = [lennard_jones_potential(np.array([ri]), 0, 0, epsilon=epsilon_new) for ri in r]
    
    fig, ax = plt.subplots()
    ax.plot(r, U_ref, 'b-', label=f'ε = {epsilon_ref:.1f}')
    ax.plot(r, U_new, 'r-', label=f'ε = {epsilon_new:.1f}')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=1.122, color='g', linestyle='--', alpha=0.3, label='Potential minimum')
    
    # Add arrows to show the deepening of the well
    ax.annotate('Deeper well\nwith ε=1.2', 
               xy=(1.122, -1.2), xytext=(1.5, -0.8),
               arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax.set_xlabel('Distance (r/σ)')
    ax.set_ylabel('Potential Energy (U/ε)')
    ax.set_title('Lennard-Jones Potential\nEffect of Increasing Attraction Strength')
    ax.legend()
    
    # Add explanation text
    fig.text(0.02, 0.02, 
             'Increasing ε deepens the attractive well\n'
             'while keeping the repulsive core unchanged',
             fontsize=10, style='italic')
    
    return fig

def plot_phase_energies():
    """Plot total energy per particle for both phases as function of epsilon."""
    epsilon_values = np.linspace(0.8, 1.4, 100)
    r_liquid = 1.14
    r_solid = 1.09
    
    # Calculate energies (already per particle)
    E_liquid = [calculate_total_energy(r_liquid, eps, 'liquid') for eps in epsilon_values]
    E_solid = [calculate_total_energy(r_solid, eps, 'solid') for eps in epsilon_values]
    
    # Print reference values
    print(f"\nEnergy values at ε = 1.0:")
    ref_idx = np.where(epsilon_values >= 1.0)[0][0]
    print(f"Liquid: {E_liquid[ref_idx]:.3f} ε/particle")
    print(f"Solid:  {E_solid[ref_idx]:.3f} ε/particle")
    print(f"Difference: {E_liquid[ref_idx] - E_solid[ref_idx]:.3f} ε/particle")
    
    print(f"\nEnergy values at ε = 1.2:")
    new_idx = np.where(epsilon_values >= 1.2)[0][0]
    print(f"Liquid: {E_liquid[new_idx]:.3f} ε/particle")
    print(f"Solid:  {E_solid[new_idx]:.3f} ε/particle")
    print(f"Difference: {E_liquid[new_idx] - E_solid[new_idx]:.3f} ε/particle")
    
    fig, ax = plt.subplots()
    ax.plot(epsilon_values, E_liquid, 'b-', label='Liquid phase')
    ax.plot(epsilon_values, E_solid, 'r-', label='Solid phase')
    
    # Mark reference and new states
    ax.plot([1.0, 1.2], [E_liquid[ref_idx], E_liquid[new_idx]], 'bo--', label='Liquid states')
    ax.plot([1.0, 1.2], [E_solid[ref_idx], E_solid[new_idx]], 'ro--', label='Solid states')
    
    # Add arrows and annotations
    ax.annotate(f'ΔE = {E_liquid[new_idx] - E_liquid[ref_idx]:.3f}ε', 
                xy=(1.1, E_liquid[new_idx]),
                xytext=(1.1, E_liquid[ref_idx]-0.2),
                arrowprops=dict(facecolor='blue', shrink=0.05))
    
    ax.set_xlabel('Interaction Strength (ε)')
    ax.set_ylabel('Energy per Particle (U/ε)')
    ax.set_title('Phase Energies vs. Interaction Strength')
    ax.legend()
    
    return fig

def plot_coexistence_curve():
    """Plot the coexistence curve from example results."""
    # Initialize the integrator with proper potentials
    epsilon_ref = 1.0
    epsilon_new = 1.2
    N = 1000
    
    u_ref = lambda coords, p, T: lennard_jones_potential(coords, p, T, epsilon=epsilon_ref)
    u_new = lambda coords, p, T: lennard_jones_potential(coords, p, T, epsilon=epsilon_new)
    
    from hamiltonian_gibbs_duhem import HamiltonianGibbsDuhemIntegrator, PhaseState
    integrator = HamiltonianGibbsDuhemIntegrator(u_ref, u_new, N)
    
    # Initial conditions
    T0 = 0.692  # melting temperature
    p0 = 5.76   # initial pressure
    
    # Setup phases with correct distances
    r_liquid = 1.14  # first peak of g(r) at melting
    r_solid = 1.09   # FCC nearest neighbor distance at melting
    
    # Calculate x_g for both phases
    x_g_liquid = calculate_phase_x_g(r_liquid, epsilon_ref, epsilon_new, 'liquid')
    x_g_solid = calculate_phase_x_g(r_solid, epsilon_ref, epsilon_new, 'solid')
    
    # Get phase properties
    phase_props = get_realistic_phase_properties(T0)
    s_liquid, v_liquid = phase_props['liquid']
    s_solid, v_solid = phase_props['solid']
    
    # Define phases
    phase1 = PhaseState(  # Liquid phase
        entropy_per_particle=s_liquid,
        volume_per_particle=v_liquid,
        x_g=x_g_liquid,
        coordinates=np.array([r_liquid])
    )
    
    phase2 = PhaseState(  # Solid phase
        entropy_per_particle=s_solid,
        volume_per_particle=v_solid,
        x_g=x_g_solid,
        coordinates=np.array([r_solid])
    )
    
    # Integrate coexistence curve
    lambda_values, T_values, p_values = integrator.integrate_coexistence_curve(
        T0=T0,
        p0=p0,
        phase1_initial=phase1,
        phase2_initial=phase2,
        lambda_range=(0.0, 1.0),
        fix_temperature=True
    )
    
    # Create single figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    
    # Plot pressure vs lambda
    ax.plot(lambda_values, p_values, 'b-', linewidth=1.5)
    ax.set_xlabel('Coupling Parameter (λ)')
    ax.set_ylabel('Coexistence Pressure (P*)')
    
    # Calculate pressure change
    dp_total = p_values[-1] - p_values[0]
    percent_change = dp_total/p0*100
    
    # Add annotations
    ax.text(0.05, 0.95, f'Initial P* = {p0:.2f}', transform=ax.transAxes)
    ax.text(0.05, 0.90, f'Final P* = {p_values[-1]:.2f}', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'ΔP/P = {percent_change:.1f}%', transform=ax.transAxes)
    
    title = f'Coexistence Pressure vs. λ\nT* = {T0:.3f}'
    ax.set_title(title)
    
    # Set y-axis limits to show full pressure range
    p_min = min(p_values) * 0.95
    p_max = max(p_values) * 1.05
    ax.set_ylim(p_min, p_max)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig

def main():
    """Create and save all plots with detailed explanations."""
    # Create and save plots one at a time
    print("Creating and saving Lennard-Jones potential plot...")
    fig1 = plot_lennard_jones_potential()
    fig1.savefig('lennard_jones_potential.png', dpi=100)
    plt.close(fig1)
    
    print("Creating and saving phase energies plot...")
    fig2 = plot_phase_energies()
    fig2.savefig('phase_energies.png', dpi=100)
    plt.close(fig2)
    
    print("Creating and saving coexistence curve plot...")
    fig3 = plot_coexistence_curve()
    fig3.savefig('coexistence_curve.png', dpi=100)
    plt.close(fig3)
    
    print("All plots have been saved successfully!")
    # Don't show interactive windows
    # plt.show()

if __name__ == "__main__":
    main() 