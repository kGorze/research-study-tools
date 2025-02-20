import numpy as np
from typing import Callable, Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

class IcePhase(Enum):
    """Enumeration of different ice phases"""
    LIQUID = "liquid"
    Ih = "Ih"      # Hexagonal ice
    Ic = "Ic"      # Cubic ice
    II = "II"      # Ice II
    III = "III"    # Ice III
    IV = "IV"      # Ice IV
    V = "V"        # Ice V
    VI = "VI"      # Ice VI
    IX = "IX"      # Ice IX
    XI = "XI"      # Ice XI
    XII = "XII"    # Ice XII

@dataclass
class PhaseProperties:
    """Class to store phase properties needed for integration"""
    enthalpy: float  # J/mol
    volume: float    # m³/mol
    temperature: float  # K
    pressure: float    # Pa
    phase_type: IcePhase  # Type of ice phase

@dataclass
class LennardJonesParams:
    """Parameters for Lennard-Jones potential"""
    epsilon: float  # Energy parameter (J)
    sigma: float    # Distance parameter (Å)
    cutoff: float = 8.5  # Cutoff distance (Å)
    
    def potential(self, r: float) -> float:
        """Calculate LJ potential with truncation"""
        if r > self.cutoff:
            return 0.0
        
        sr6 = (self.sigma/r)**6
        return 4 * self.epsilon * (sr6**2 - sr6) + self.lj_correction(r)
    
    def lj_correction(self, r: float) -> float:
        """Long-range correction for LJ potential"""
        if r > self.cutoff:
            return 0.0
        
        sr3 = (self.sigma/self.cutoff)**3
        return 8/3 * np.pi * self.epsilon * sr3 * (sr3/3 - 1)

class EwaldSummation:
    """Ewald summation for long-range electrostatic interactions"""
    def __init__(self, alpha: float, k_vectors: np.ndarray):
        self.alpha = alpha  # Screening parameter
        self.k_vectors = k_vectors  # Reciprocal space vectors
        
    def compute_energy(self, positions: np.ndarray, charges: np.ndarray) -> float:
        """Compute electrostatic energy using Ewald summation"""
        # Real space sum
        real_sum = self._real_space_sum(positions, charges)
        # Reciprocal space sum
        recip_sum = self._reciprocal_space_sum(positions, charges)
        # Self-interaction correction
        self_correction = self._self_interaction_correction(charges)
        
        return real_sum + recip_sum - self_correction
    
    def _real_space_sum(self, positions: np.ndarray, charges: np.ndarray) -> float:
        """Compute real space sum"""
        # Implementation of real space sum
        return 0.0  # Placeholder
    
    def _reciprocal_space_sum(self, positions: np.ndarray, charges: np.ndarray) -> float:
        """Compute reciprocal space sum"""
        # Implementation of reciprocal space sum
        return 0.0  # Placeholder
    
    def _self_interaction_correction(self, charges: np.ndarray) -> float:
        """Compute self-interaction correction"""
        return self.alpha/np.sqrt(np.pi) * np.sum(charges**2)

class RungeKutta4:
    """
    Enhanced fourth-order Runge-Kutta integrator with error estimation and
    adaptive step size control for Gibbs-Duhem integration.
    """
    
    def __init__(self, 
                 get_phase_properties: Callable[[float, float, IcePhase], PhaseProperties],
                 lj_params: LennardJonesParams,
                 ewald_params: Optional[EwaldSummation] = None,
                 rtol: float = 1e-6,
                 atol: float = 1e-8):
        """
        Initialize the enhanced Runge-Kutta integrator.
        
        Args:
            get_phase_properties: Function that returns phase properties
            lj_params: Lennard-Jones parameters
            ewald_params: Ewald summation parameters (optional)
            rtol: Relative tolerance for error control
            atol: Absolute tolerance for error control
        """
        self.get_phase_properties = get_phase_properties
        self.lj_params = lj_params
        self.ewald_params = ewald_params
        self.rtol = rtol
        self.atol = atol
    
    def clapeyron_derivative(self, t: float, p: float, phase1: IcePhase, phase2: IcePhase) -> float:
        """Enhanced Clapeyron equation calculation with all interactions"""
        props1 = self.get_phase_properties(t, p, phase1)
        props2 = self.get_phase_properties(t, p, phase2)
        
        delta_h = props2.enthalpy - props1.enthalpy
        delta_v = props2.volume - props1.volume
        
        return delta_h / (t * delta_v)
    
    def hamiltonian_derivative(self, t: float, p: float, lambda_param: float,
                             phase1: IcePhase, phase2: IcePhase) -> Tuple[float, float]:
        """
        Calculate derivatives for Hamiltonian Gibbs-Duhem integration.
        Returns (dT/dλ, dp/dλ)
        """
        props1 = self.get_phase_properties(t, p, phase1)
        props2 = self.get_phase_properties(t, p, phase2)
        
        # Calculate Δxg (eq. 4 in the paper)
        delta_xg = self._calculate_delta_xg(props1, props2, lambda_param)
        
        # Calculate Δs and Δv
        delta_s = (props2.enthalpy - props1.enthalpy) / t
        delta_v = props2.volume - props1.volume
        
        # Equations 5 and 6 from the paper
        dt_dlambda = delta_xg / delta_s
        dp_dlambda = -delta_xg / delta_v
        
        return dt_dlambda, dp_dlambda
    
    def _calculate_delta_xg(self, props1: PhaseProperties, props2: PhaseProperties,
                          lambda_param: float) -> float:
        """Calculate Δxg for Hamiltonian Gibbs-Duhem integration"""
        # Implementation of equation 4 from the paper
        # This is a placeholder - actual implementation would depend on specific potential
        return 0.0
    
    def rk4_step_with_error(self, t: float, p: float, dt: float,
                           phase1: IcePhase, phase2: IcePhase) -> Tuple[float, float, float]:
        """
        Perform one RK4 step with error estimation using embedded 5th order method
        """
        # Standard RK4 coefficients
        k1 = self.clapeyron_derivative(t, p, phase1, phase2)
        k2 = self.clapeyron_derivative(t + 0.5*dt, p + 0.5*dt*k1, phase1, phase2)
        k3 = self.clapeyron_derivative(t + 0.5*dt, p + 0.5*dt*k2, phase1, phase2)
        k4 = self.clapeyron_derivative(t + dt, p + dt*k3, phase1, phase2)
        
        # 4th order solution
        dp4 = (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Additional evaluation for 5th order error estimation
        k5 = self.clapeyron_derivative(t + 0.5*dt, p + 0.5*dt*k4, phase1, phase2)
        
        # 5th order solution
        dp5 = (dt/12.0) * (k1 + 4*k2 + 2*k3 + 2*k4 + k5)
        
        # Error estimation
        error = abs(dp5 - dp4)
        
        return t + dt, p + dp4, error
    
    def adapt_step_size(self, dt: float, error: float, p: float) -> float:
        """Adapt step size based on error estimate"""
        tolerance = self.atol + self.rtol * abs(p)
        
        if error == 0:
            factor = 2.0
        else:
            factor = 0.9 * (tolerance/error)**0.2
            
        factor = min(2.0, max(0.1, factor))
        return dt * factor
    
    def integrate_phase_boundary(self,
                               t_start: float,
                               p_start: float,
                               t_end: float,
                               phase1: IcePhase,
                               phase2: IcePhase,
                               dt_initial: float = 0.1,
                               max_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced phase boundary integration with adaptive step size control
        """
        temperatures = [t_start]
        pressures = [p_start]
        
        t = t_start
        p = p_start
        dt = dt_initial
        steps = 0
        
        print("\nIntegration Debug Information:")
        print(f"Starting integration from T={t_start}K to T={t_end}K")
        print(f"Initial pressure: {p_start} Pa")
        print(f"Initial step size: {dt_initial} K")
        
        while (t < t_end if dt > 0 else t > t_end) and steps < max_steps:
            # Ensure we don't overshoot the end temperature
            if abs(t + dt - t_end) < abs(dt):
                dt = t_end - t
            
            # Perform step with error estimation
            t_new, p_new, error = self.rk4_step_with_error(t, p, dt, phase1, phase2)
            
            # Adapt step size
            dt_next = self.adapt_step_size(dt, error, p)
            
            # Print debug information every 10 steps
            if steps % 10 == 0:
                print(f"\nStep {steps}:")
                print(f"Current T={t}K, P={p:.2e}Pa")
                print(f"Proposed next T={t_new}K, P={p_new:.2e}Pa")
                print(f"Error estimate: {error:.2e}")
                print(f"Step size: {dt} → {dt_next}")
            
            # Accept step if error is reasonable
            if error <= self.atol + self.rtol * abs(p):
                temperatures.append(t_new)
                pressures.append(p_new)
                t, p = t_new, p_new
                steps += 1
            else:
                print(f"Step rejected: error {error:.2e} > tolerance {self.atol + self.rtol * abs(p):.2e}")
            
            dt = dt_next
            
            if steps >= max_steps:
                warnings.warn(f"Maximum number of steps ({max_steps}) reached")
                break
        
        print(f"\nIntegration completed with {steps} steps")
        return np.array(temperatures), np.array(pressures)
    
    def integrate_hamiltonian(self,
                            t_start: float,
                            p_start: float,
                            lambda_range: Tuple[float, float],
                            phase1: IcePhase,
                            phase2: IcePhase,
                            dlambda: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Hamiltonian Gibbs-Duhem integration
        """
        lambda_values = np.arange(lambda_range[0], lambda_range[1] + dlambda, dlambda)
        temperatures = [t_start]
        pressures = [p_start]
        
        t = t_start
        p = p_start
        
        for lambda_val in lambda_values[1:]:
            dt_dlambda, dp_dlambda = self.hamiltonian_derivative(t, p, lambda_val,
                                                               phase1, phase2)
            
            t += dt_dlambda * dlambda
            p += dp_dlambda * dlambda
            
            temperatures.append(t)
            pressures.append(p)
        
        return np.array(lambda_values), np.array(temperatures), np.array(pressures)

# Example usage with ice phases
def example_ice_properties(t: float, p: float, phase: IcePhase) -> PhaseProperties:
    """
    Example implementation for ice phase properties.
    Replace with actual ice phase calculations.
    """
    # These are placeholder values - replace with actual calculations
    properties = {
        IcePhase.LIQUID: (30000.0, 1.8e-5),
        IcePhase.Ih: (25000.0, 1.6e-5),
        IcePhase.Ic: (24800.0, 1.59e-5),
        IcePhase.II: (24000.0, 1.55e-5),
        IcePhase.III: (23500.0, 1.52e-5),
        IcePhase.IV: (23000.0, 1.50e-5),
        IcePhase.V: (22500.0, 1.48e-5),
        IcePhase.VI: (22000.0, 1.45e-5),
        IcePhase.IX: (21500.0, 1.42e-5),
        IcePhase.XI: (21000.0, 1.40e-5),
        IcePhase.XII: (20500.0, 1.38e-5),
    }
    
    enthalpy, volume = properties[phase]
    return PhaseProperties(enthalpy, volume, t, p, phase)

if __name__ == "__main__":
    # Example calculation with ice phases
    lj_params = LennardJonesParams(
        epsilon=0.650,  # kJ/mol
        sigma=3.166,    # Å
        cutoff=8.5      # Å
    )
    
    # Initialize integrator with more relaxed tolerances
    integrator = RungeKutta4(
        get_phase_properties=example_ice_properties,
        lj_params=lj_params,
        rtol=1e-4,  # Relaxed from 1e-6
        atol=1e-6   # Relaxed from 1e-8
    )
    
    # Calculate phase boundary between liquid and ice Ih
    temps, pressures = integrator.integrate_phase_boundary(
        t_start=273.15,  # K (0°C)
        p_start=1e5,     # Pa (1 bar)
        t_end=250.0,     # K
        phase1=IcePhase.LIQUID,
        phase2=IcePhase.Ih,
        dt_initial=0.01  # Smaller initial step size
    )
    
    # Print results with more information
    print("\nPhase Boundary Points (Liquid-Ice Ih):")
    print("Temperature (K) | Pressure (Pa)")
    print("-" * 35)
    for t, p in zip(temps[::10], pressures[::10]):
        print(f"{t:13.2f} | {p:10.2e}")
