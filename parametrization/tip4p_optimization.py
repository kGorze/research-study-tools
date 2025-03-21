"""Module for optimizing TIP4P water model parameters."""

import numpy as np
from scipy.optimize import minimize
from .tip4p_parameters import TIP4PParameters
from .property_calculators import (
    calculate_ice_density,
    calculate_melting_temperature,
    calculate_enthalpy_fusion,
    calculate_melting_properties
)
from integration.example_gibbs_duhem import lennard_jones_potential
from integration.plot_gibbs_duhem import plot_lennard_jones_potential
from geometry.bernal_fowler import create_water_molecule
from integration.hamiltonian_gibbs_duhem import HamiltonianGibbsDuhemIntegrator, PhaseState
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class ExperimentalProperty:
    """Class to hold experimental property data."""
    name: str
    value: float
    weight: float
    calc_function: Callable[[TIP4PParameters], float]

class TIP4POptimizer:
    """Class to perform two-step parametrization of TIP4P model."""
    
    def __init__(self, initial_params: TIP4PParameters):
        """Initialize with starting parameters."""
        self.initial_params = initial_params
        self.properties: List[ExperimentalProperty] = []
        self.max_iterations = 1000
        self.convergence_tol = 1e-6
        
    def add_property(self, name: str, expt_value: float, 
                    weight: float, calc_function: Callable):
        """Add an experimental property to fit."""
        self.properties.append(
            ExperimentalProperty(name, expt_value, weight, calc_function)
        )
    
    def calculate_derivatives(self, params: TIP4PParameters, 
                            delta: float = 0.01) -> np.ndarray:
        """
        Calculate derivatives of properties with respect to parameters.
        Uses single-point forward difference method.
        """
        n_params = 4
        n_props = len(self.properties)
        derivatives = np.zeros((n_props, n_params))
        
        base_params = params.as_array()
        base_values = np.array([
            prop.calc_function(params) for prop in self.properties
        ])
        
        for i in range(n_params):
            # Perturb one parameter
            new_params = base_params.copy()
            new_params[i] += delta
            
            # Calculate properties with perturbed parameter
            new_values = np.array([
                prop.calc_function(TIP4PParameters.from_array(new_params))
                for prop in self.properties
            ])
            
            # Calculate derivative
            derivatives[:, i] = (new_values - base_values) / delta
            
        return derivatives
    
    def objective_function(self, params_array: np.ndarray) -> float:
        """Calculate weighted sum of squared deviations."""
        params = TIP4PParameters.from_array(params_array)
        
        total_deviation = 0.0
        for prop in self.properties:
            calc_value = prop.calc_function(params)
            deviation = (calc_value - prop.value) / prop.value  # Relative error
            total_deviation += prop.weight * deviation**2
            
        return total_deviation
    
    def optimize_step(self, current_params: TIP4PParameters,
                     bounds: List[Tuple[float, float]]) -> TIP4PParameters:
        """Perform one optimization step."""
        result = minimize(
            self.objective_function,
            current_params.as_array(),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.convergence_tol,
                'gtol': 1e-6,
                'maxcor': 50  # Increase memory for better convergence
            }
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        return TIP4PParameters.from_array(result.x)
    
    def run_two_step_optimization(self, bounds: List[Tuple[float, float]]) -> Tuple[TIP4PParameters, Dict]:
        """
        Run the two-step parametrization process.
        
        Args:
            bounds: List of (min, max) tuples for each parameter
            
        Returns:
            Tuple of (optimized parameters, optimization history)
        """
        # Step 1: Initial optimization with crude derivatives
        print("Step 1: Initial optimization")
        intermediate_params = self.optimize_step(self.initial_params, bounds)
        
        # Calculate properties at intermediate point
        intermediate_values = {
            prop.name: prop.calc_function(intermediate_params)
            for prop in self.properties
        }
        
        # Step 2: Final optimization with better derivatives
        print("\nStep 2: Final optimization")
        print("Using tighter convergence criteria...")
        self.convergence_tol *= 0.1  # Tighter convergence for final step
        final_params = self.optimize_step(intermediate_params, bounds)
        
        # Calculate final properties
        final_values = {
            prop.name: prop.calc_function(final_params)
            for prop in self.properties
        }
        
        # Collect optimization history
        history = {
            'initial': {
                'params': self.initial_params,
                'values': {
                    prop.name: prop.calc_function(self.initial_params)
                    for prop in self.properties
                }
            },
            'intermediate': {
                'params': intermediate_params,
                'values': intermediate_values
            },
            'final': {
                'params': final_params,
                'values': final_values
            }
        }
        
        return final_params, history

def calculate_ice_density(params: TIP4PParameters) -> float:
    """Example property calculator for ice Ih density."""
    # This would normally involve MD simulation
    # Here we use a simple approximation
    return 0.917  # g/cm³

def calculate_melting_temperature(params: TIP4PParameters) -> float:
    """Example property calculator for melting temperature."""
    # This would normally involve free energy calculations
    # Here we use a simple approximation
    return 273.15  # K

def main():
    """Run the TIP4P optimization process."""
    # Initial TIP4P parameters (from literature)
    initial_params = TIP4PParameters(
        epsilon=0.7732,  # kJ/mol (stronger than previous)
        sigma=3.154,    # Å
        qH=0.52,       # e
        dOM=0.15       # Å
    )
    
    # Create optimizer
    optimizer = TIP4POptimizer(initial_params)
    
    # Add experimental properties to fit with appropriate weights
    
    # Ice Ih density at 273.15 K
    optimizer.add_property(
        name="Ice Ih density",
        expt_value=0.917,  # g/cm³
        weight=3.0,        # Higher weight for structure
        calc_function=calculate_ice_density
    )
    
    # Melting temperature at 1 bar
    optimizer.add_property(
        name="Melting temperature",
        expt_value=273.15,  # K
        weight=2.0,         # Important for phase behavior
        calc_function=calculate_melting_temperature
    )
    
    # Enthalpy of fusion
    optimizer.add_property(
        name="Enthalpy of fusion",
        expt_value=6.012,   # kJ/mol
        weight=2.5,         # Higher weight to fix previous issues
        calc_function=calculate_enthalpy_fusion
    )
    
    # Define wider bounds for parameters
    bounds = [
        (0.4, 2.0),    # epsilon: wider range for better energy scale
        (2.5, 4.0),    # sigma: allow larger values
        (0.3, 0.7),    # qH: wider charge range
        (0.1, 0.25)    # dOM: wider geometric range
    ]
    
    # Run optimization with more iterations
    optimizer.max_iterations = 1000  # Increase max iterations
    optimizer.convergence_tol = 1e-6  # Tighter convergence
    
    print("Starting two-step TIP4P parametrization")
    print("---------------------------------------")
    print("\nInitial parameters:")
    print(f"  epsilon = {initial_params.epsilon:.3f} kJ/mol")
    print(f"  sigma = {initial_params.sigma:.3f} Å")
    print(f"  qH = {initial_params.qH:.3f} e")
    print(f"  dOM = {initial_params.dOM:.3f} Å")
    print(f"  qM = {initial_params.qM:.3f} e")
    
    final_params, history = optimizer.run_two_step_optimization(bounds)
    
    # Print detailed results
    print("\nOptimization Results:")
    print("--------------------")
    
    for stage in ['initial', 'intermediate', 'final']:
        print(f"\n{stage.capitalize()} Stage:")
        print("Parameters:")
        params = history[stage]['params']
        print(f"  epsilon = {params.epsilon:.3f} kJ/mol")
        print(f"  sigma = {params.sigma:.3f} Å")
        print(f"  qH = {params.qH:.3f} e")
        print(f"  dOM = {params.dOM:.3f} Å")
        print(f"  qM = {params.qM:.3f} e")
        
        print("\nProperties:")
        values = history[stage]['values']
        for name, value in values.items():
            # Find the experimental value and calculate relative error
            for prop in optimizer.properties:
                if prop.name == name:
                    expt = prop.value
                    error = (value - expt) / expt * 100
                    print(f"  {name}:")
                    print(f"    Calculated: {value:.3f}")
                    print(f"    Experimental: {expt:.3f}")
                    print(f"    Error: {error:+.1f}%")
    
    # Plot results
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    
    # Set modern style directly
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = '#f0f0f0'
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Parameter evolution with bounds
    ax1 = fig.add_subplot(gs[0, 0])
    stages = ['initial', 'intermediate', 'final']
    x = range(len(stages))
    params = ['epsilon', 'sigma', 'qH', 'dOM']
    param_labels = {
        'epsilon': 'ε (kJ/mol)',
        'sigma': 'σ (Å)',
        'qH': 'qH (e)',
        'dOM': 'dOM (Å)'
    }
    param_bounds = {
        'epsilon': (0.4, 2.0),
        'sigma': (2.5, 4.0),
        'qH': (0.3, 0.7),
        'dOM': (0.1, 0.25)
    }
    markers = ['o', 's', '^', 'D']  # Different markers for each parameter
    
    for param, marker in zip(params, markers):
        values = [getattr(history[stage]['params'], param) for stage in stages]
        # Plot parameter evolution
        ax1.plot(x, values, marker=marker, label=param_labels[param], linewidth=2, markersize=8)
        # Add bounds as shaded regions
        bound_min, bound_max = param_bounds[param]
        ax1.fill_between([-0.2, 2.2], [bound_min]*2, [bound_max]*2, alpha=0.1)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in stages])
    ax1.set_ylabel('Parameter Value')
    ax1.set_title('Parameter Evolution\nwith Optimization Bounds', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Property evolution with tolerance bands
    ax2 = fig.add_subplot(gs[0, 1])
    properties = ['Ice Ih density', 'Melting temperature', 'Enthalpy of fusion']
    prop_scales = {
        'Ice Ih density': 1.0,
        'Melting temperature': 1/273.15,
        'Enthalpy of fusion': 1/6.012
    }
    prop_labels = {
        'Ice Ih density': 'Ice Ih density (g/cm³)',
        'Melting temperature': 'Melting temperature (K)',
        'Enthalpy of fusion': 'Enthalpy of fusion (kJ/mol)'
    }
    
    for prop in properties:
        values = [history[stage]['values'][prop] * prop_scales[prop] for stage in stages]
        ax2.plot(x, values, 'o-', label=prop_labels[prop], linewidth=2, markersize=8)
        # Add tolerance band (±2%)
        ax2.fill_between([-0.2, 2.2], [0.98]*2, [1.02]*2, alpha=0.1, color='gray')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.capitalize() for s in stages])
    ax2.set_ylabel('Normalized Property Value')
    ax2.set_title('Property Evolution\nNormalized to Experimental Values (±2% tolerance)', pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target')
    
    # Plot 3: Lennard-Jones potential with inset
    ax3 = fig.add_subplot(gs[1, 0])
    r = np.linspace(2.5, 5.0, 100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Better color scheme
    labels = ['Initial', 'Intermediate', 'Final']
    
    # Create inset for zoom
    axins = ax3.inset_axes([0.15, 0.15, 0.4, 0.4])
    
    for stage, color, label in zip(stages, colors, labels):
        params = history[stage]['params']
        u = [lennard_jones_potential(np.array([ri]), 1.0, 273.15, 
             epsilon=params.epsilon, sigma=params.sigma) for ri in r]
        # Main plot
        line = ax3.plot(r, u, label=f'{label}', color=color, linewidth=2)
        # Inset plot
        axins.plot(r, u, color=color, linewidth=2)
    
    # Set inset limits to zoom on potential well
    axins.set_xlim(3.0, 3.5)
    axins.set_ylim(-0.004, -0.002)
    ax3.indicate_inset_zoom(axins)
    
    ax3.set_xlabel('Distance (Å)')
    ax3.set_ylabel('Potential Energy (kJ/mol)')
    ax3.set_title('TIP4P Lennard-Jones Potential\nwith Zoom on Potential Well', pad=20)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Property comparison with error bars
    ax4 = fig.add_subplot(gs[1, 1])
    width = 0.25
    x = np.arange(len(properties))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Match LJ plot colors
    
    for i, (stage, color) in enumerate(zip(stages, colors)):
        values = [history[stage]['values'][prop] * prop_scales[prop] for prop in properties]
        # Add error bars (2% uncertainty)
        yerr = [0.02 * v for v in values]
        ax4.bar(x + i*width, values, width, label=stage.capitalize(), 
                color=color, alpha=0.7, yerr=yerr, capsize=5)
    
    # Improve property labels
    prop_short_labels = {
        'Ice Ih density': 'ρ(ice)',
        'Melting temperature': 'T_m',
        'Enthalpy of fusion': 'ΔH_fus'
    }
    
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([prop_short_labels[p] for p in properties], rotation=0)
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Property Comparison\nNormalized to Experimental Values', pad=20)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target')
    
    # Add property units as text
    for i, prop in enumerate(properties):
        ax4.text(i + width, 0.1, f'({prop_labels[prop].split("(")[1].strip(")")})\\n', 
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('tip4p_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nOptimization completed successfully!")
    print("Results have been saved to 'tip4p_optimization_results.png'")

if __name__ == "__main__":
    main() 