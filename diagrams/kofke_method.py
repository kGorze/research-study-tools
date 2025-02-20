import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt

class KofkeMethod:
    """
    Implementation of Kofke's method (Gibbs-Duhem integration) for calculating phase diagrams
    by integrating the Clapeyron equation:
    dp/dT = Δh/(TΔv)
    where:
    - Δh is the enthalpy change
    - Δv is the volume change
    - T is temperature
    - p is pressure
    """
    
    def __init__(self, delta_h_func: Callable[[float, float], float],
                 delta_v_func: Callable[[float, float], float]):
        """
        Initialize the Kofke method solver.
        
        Args:
            delta_h_func: Function that returns Δh given T and p
            delta_v_func: Function that returns Δv given T and p
        """
        self.delta_h_func = delta_h_func
        self.delta_v_func = delta_v_func
    
    def clapeyron_equation(self, t: float, p: float) -> float:
        """
        The Clapeyron equation dp/dT = Δh/(TΔv).
        
        Args:
            t: Temperature
            p: Pressure
            
        Returns:
            dp/dT value at the given point
        """
        delta_h = self.delta_h_func(t, p)
        delta_v = self.delta_v_func(t, p)
        
        return delta_h / (t * delta_v)
    
    def integrate_phase_boundary(self, t_start: float, p_start: float,
                               t_range: Tuple[float, float],
                               t_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the Clapeyron equation to get the phase boundary.
        
        Args:
            t_start: Starting temperature
            p_start: Starting pressure
            t_range: (min_temp, max_temp) range for integration
            t_points: Number of temperature points
            
        Returns:
            Tuple of (temperatures, pressures) arrays defining the phase boundary
        """
        def deriv(t: float, p: float) -> float:
            return self.clapeyron_equation(t, p)
        
        # Solve the ODE
        solution = solve_ivp(
            fun=lambda t, y: deriv(t, y[0]),
            t_span=t_range,
            y0=[p_start],
            t_eval=np.linspace(t_range[0], t_range[1], t_points),
            method='RK45',
            rtol=1e-8
        )
        
        return solution.t, solution.y[0]
    
    def plot_phase_diagram(self, temperatures: np.ndarray, pressures: np.ndarray,
                         xlabel: str = 'Temperature', ylabel: str = 'Pressure',
                         title: str = 'Phase Diagram'):
        """
        Plot the calculated phase diagram.
        
        Args:
            temperatures: Array of temperature values
            pressures: Array of pressure values
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(temperatures, pressures, 'b-', label='Phase boundary')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

# Example usage:
def example_delta_h(t: float, p: float) -> float:
    """Example enthalpy change function (should be replaced with actual system)"""
    return 30000.0  # Example constant value in J/mol

def example_delta_v(t: float, p: float) -> float:
    """Example volume change function (should be replaced with actual system)"""
    return 1e-5  # Example constant value in m³/mol

if __name__ == "__main__":
    # Create instance with example functions
    kofke = KofkeMethod(example_delta_h, example_delta_v)
    
    # Calculate phase boundary
    temps, pressures = kofke.integrate_phase_boundary(
        t_start=300.0,  # Starting at 300K
        p_start=1e5,    # Starting at 1 bar
        t_range=(300.0, 500.0)  # Calculate from 300K to 500K
    )
    
    # Plot the results
    kofke.plot_phase_diagram(temps, pressures)
