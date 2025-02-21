import numpy as np
from hamiltonian_gibbs_duhem import HamiltonianGibbsDuhemIntegrator, PhaseState

def lennard_jones_potential(coordinates: np.ndarray, p: float, T: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """
    Simple Lennard-Jones potential for demonstration.
    U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
    
    Note: For a single pair, the attractive well depth is ε at r ≈ 1.122σ
    
    Args:
        coordinates: Array of distances in reduced units (σ)
        p: Pressure in reduced units (ε/σ³)
        T: Temperature in reduced units (k_B T/ε)
        epsilon: Energy scale in reduced units
        sigma: Length scale in reduced units
        
    Returns:
        Potential energy in reduced units (ε)
    """
    r = np.linalg.norm(coordinates)
    if r < 0.1:  # Avoid singularity at r=0
        return 1e10
    sr6 = (1.0/r)**6  # Using reduced units where σ=1
    sr12 = sr6*sr6
    return 4.0 * epsilon * (sr12 - sr6)

def calculate_total_energy(r: float, epsilon: float, phase: str) -> float:
    """Calculate total energy contribution from all neighbor shells."""
    if phase == 'solid':
        # FCC structure shells with proper weighting
        # r is the nearest neighbor distance in reduced units (σ)
        neighbor_shells = [
            (12, r),              # 12 nearest neighbors at r
            (6, r * 1.414),       # 6 next-nearest at √2r
            (24, r * 1.732),      # 24 at √3r
            (12, r * 2.000),      # 12 at 2r
            (24, r * 2.236),      # 24 at √5r
            (8, r * 2.449)        # 8 at √6r
        ]
    else:  # liquid
        # Liquid structure based on radial distribution function
        # r is the first peak distance in reduced units (σ)
        # Coordination numbers from MD simulations at melting
        neighbor_shells = [
            (12.5, r),            # First coordination shell (g(r) first peak)
            (45.0, r * 1.633),    # Second shell (g(r) second peak)
            (85.0, r * 2.171)     # Third shell (g(r) third peak)
        ]
    
    total_energy = 0.0
    for n_neighbors, distance in neighbor_shells:
        if distance > 2.5:  # Cutoff for LJ
            continue
        coords = np.array([distance])
        energy_per_pair = lennard_jones_potential(coords, 0, 0, epsilon=epsilon)
        total_energy += 0.5 * n_neighbors * energy_per_pair  # Factor 0.5 to avoid double counting
    
    return total_energy

def calculate_phase_x_g(r: float, epsilon_ref: float, epsilon_new: float, phase: str) -> float:
    """
    Calculate x_g as the difference in total energy between new and reference potentials.
    Now properly accounts for all neighbor shell contributions.
    """
    # Calculate total energy for both potentials with their respective epsilon values
    U_ref = calculate_total_energy(r, epsilon_ref, phase)
    U_new = calculate_total_energy(r, epsilon_new, phase)
    
    # x_g is the energy difference per particle
    return U_new - U_ref

def get_realistic_phase_properties(T: float):
    """
    Get realistic entropy and volume differences for LJ system.
    Values from molecular dynamics simulations at melting.
    """
    # Entropy values at melting (in reduced units)
    # Using more realistic values that don't scale directly with T
    s_solid = -2.2    # More ordered structure
    s_liquid = -1.8   # More disorder
    
    # Volumes at melting (in reduced units)
    v_solid = 1.0     # Reference volume (FCC)
    v_liquid = 1.08   # ~8% expansion typical for LJ melting
    
    return {
        'solid': (s_solid, v_solid),
        'liquid': (s_liquid, v_liquid)
    }

def main():
    # Define reference and new potentials with different epsilon values
    epsilon_ref = 1.0
    epsilon_new = 1.2  # 20% stronger attraction
    
    u_ref = lambda coords, p, T: lennard_jones_potential(coords, p, T, epsilon=epsilon_ref)
    u_new = lambda coords, p, T: lennard_jones_potential(coords, p, T, epsilon=epsilon_new)
    
    # Initialize the integrator
    N = 1000  # number of particles
    integrator = HamiltonianGibbsDuhemIntegrator(u_ref, u_new, N)
    
    # Use literature values for coexistence point
    T0 = 0.692  # melting temperature in reduced units
    p0 = 5.76   # melting pressure in reduced units
    
    # Typical nearest-neighbor distances in reduced units
    r_liquid = 1.14  # first peak of g(r) at melting
    r_solid = 1.09   # FCC nearest neighbor distance at melting
    
    # Create representative configurations
    coords_liquid = np.array([r_liquid])
    coords_solid = np.array([r_solid])
    
    # Calculate x_g for both phases
    x_g_liquid = calculate_phase_x_g(r_liquid, epsilon_ref, epsilon_new, 'liquid')
    x_g_solid = calculate_phase_x_g(r_solid, epsilon_ref, epsilon_new, 'solid')
    
    # Get realistic phase properties
    phase_props = get_realistic_phase_properties(T0)
    s_liquid, v_liquid = phase_props['liquid']
    s_solid, v_solid = phase_props['solid']
    
    # Define phases
    phase1 = PhaseState(  # Liquid phase
        entropy_per_particle=s_liquid,
        volume_per_particle=v_liquid,
        x_g=x_g_liquid,
        coordinates=coords_liquid
    )
    
    phase2 = PhaseState(  # Solid phase
        entropy_per_particle=s_solid,
        volume_per_particle=v_solid,
        x_g=x_g_solid,
        coordinates=coords_solid
    )
    
    lambda_range = (0.0, 1.0)
    
    # Integrate the coexistence curve
    lambda_values, T_values, p_values = integrator.integrate_coexistence_curve(
        T0=T0,
        p0=p0,
        phase1_initial=phase1,
        phase2_initial=phase2,
        lambda_range=lambda_range,
        fix_temperature=True
    )
    
    # Calculate changes
    total_dp = p_values[-1] - p_values[0]
    
    # Print results
    print("Hamiltonian Gibbs-Duhem Integration Results")
    print("------------------------------------------")
    print("System: Lennard-Jones potential")
    print(f"Reference ε: {epsilon_ref:.3f}")
    print(f"New ε: {epsilon_new:.3f} (20% stronger attraction)")
    print(f"\nPhase configurations:")
    print(f"Liquid: r={r_liquid:.3f}σ (with RDF-based shells)")
    print(f"Solid: r={r_solid:.3f}σ (FCC structure)")
    print(f"\nEnergies per particle:")
    print(f"Liquid (ref): {calculate_total_energy(r_liquid, epsilon_ref, 'liquid')/N:.6f}")
    print(f"Liquid (new): {calculate_total_energy(r_liquid, epsilon_new, 'liquid')/N:.6f}")
    print(f"Solid (ref): {calculate_total_energy(r_solid, epsilon_ref, 'solid')/N:.6f}")
    print(f"Solid (new): {calculate_total_energy(r_solid, epsilon_new, 'solid')/N:.6f}")
    print(f"\nPhase properties:")
    print(f"Liquid: s={phase1.entropy_per_particle:.6f}, v={phase1.volume_per_particle:.6f}, x_g={phase1.x_g:.6f}")
    print(f"Solid: s={phase2.entropy_per_particle:.6f}, v={phase2.volume_per_particle:.6f}, x_g={phase2.x_g:.6f}")
    print(f"\nPressure integration:")
    print(f"Initial pressure: {p0:.6f}")
    print(f"Final pressure: {p_values[-1]:.6f}")
    print(f"Pressure values: {p_values}")
    print(f"Lambda values: {lambda_values}")
    print(f"\nCoexistence curve at T = {T0:.3f}:")
    print(f"{'λ':>10} {'Temperature':>15} {'Pressure':>15}")
    print("-" * 40)
    for l, t, p in zip(lambda_values, T_values, p_values):
        print(f"{l:10.4f} {t:15.4f} {p:15.4f}")
    print("\nSummary:")
    print(f"Total pressure change: {total_dp:.6f}")
    print(f"Relative pressure change: {(total_dp/p_values[0]*100):.2f}%")

if __name__ == "__main__":
    main() 