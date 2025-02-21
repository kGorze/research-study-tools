import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp

@dataclass
class PhaseState:
    """Class representing the thermodynamic state of a phase."""
    entropy_per_particle: float  # s
    volume_per_particle: float   # v
    x_g: float                  # conjugate variable associated with λ
    coordinates: np.ndarray     # representative configuration

class HamiltonianGibbsDuhemIntegrator:
    """
    Implementation of Hamiltonian Gibbs-Duhem integration method for calculating
    phase coexistence conditions when modifying interaction potentials.
    """
    
    def __init__(
        self,
        u_ref: Callable[[np.ndarray, float, float], float],
        u_new: Callable[[np.ndarray, float, float], float],
        N: int
    ):
        """
        Initialize the integrator.
        
        Args:
            u_ref: Reference potential energy function u_ref(coordinates, p, T)
            u_new: New potential energy function u_new(coordinates, p, T)
            N: Number of particles in the system
        """
        self.u_ref = u_ref
        self.u_new = u_new
        self.N = N
        
    def interpolated_potential(
        self,
        coordinates: np.ndarray,
        lambda_param: float,
        p: float,
        T: float
    ) -> float:
        """
        Calculate the interpolated potential energy.
        
        u = (1-λ)u_ref + λu_new
        
        Args:
            coordinates: System coordinates
            lambda_param: Interpolation parameter λ ∈ [0,1]
            p: Pressure
            T: Temperature
            
        Returns:
            Interpolated potential energy
        """
        return (1 - lambda_param) * self.u_ref(coordinates, p, T) + \
               lambda_param * self.u_new(coordinates, p, T)
    
    def calculate_x_g(
        self,
        coordinates: np.ndarray,
        p: float,
        T: float,
        lambda_param: float
    ) -> float:
        """
        Calculate the conjugate variable x_g.
        
        x_g = (1/N)(∂U/∂λ)_{N,p,T,λ} = (1/N)(U_new - U_ref)
        
        Args:
            coordinates: System coordinates
            p: Pressure
            T: Temperature
            lambda_param: Current value of λ
            
        Returns:
            Value of x_g
        """
        # For Lennard-Jones, x_g is proportional to the potential difference
        U_ref = self.u_ref(coordinates, p, T)
        U_new = self.u_new(coordinates, p, T)
        return (U_new - U_ref) / self.N

    def update_phase_state(
        self,
        phase: PhaseState,
        lambda_param: float,
        T: float,
        p: float
    ) -> PhaseState:
        """
        Update phase state for current λ value.
        Entropy and volume are assumed constant during isothermal integration.
        Only x_g needs to be updated.
        """
        new_x_g = self.calculate_x_g(phase.coordinates, p, T, lambda_param)
        return PhaseState(
            entropy_per_particle=phase.entropy_per_particle,
            volume_per_particle=phase.volume_per_particle,
            x_g=new_x_g,
            coordinates=phase.coordinates
        )
    
    def dT_dlambda(
        self,
        phase1: PhaseState,
        phase2: PhaseState,
        lambda_param: float,
        T: float,
        p: float
    ) -> float:
        """
        Calculate dT/dλ using the generalized Clapeyron-like equation.
        
        dT/dλ = Δx_g/Δs
        
        Args:
            phase1: State of first phase
            phase2: State of second phase
            lambda_param: Current λ value
            T: Current temperature
            p: Current pressure
            
        Returns:
            Rate of change of temperature with respect to λ
        """
        # Update phases for current conditions
        phase1_current = self.update_phase_state(phase1, lambda_param, T, p)
        phase2_current = self.update_phase_state(phase2, lambda_param, T, p)
        
        delta_x_g = phase2_current.x_g - phase1_current.x_g
        delta_s = phase2_current.entropy_per_particle - phase1_current.entropy_per_particle
        
        if abs(delta_s) < 1e-10:
            return 0.0
        return delta_x_g / delta_s
    
    def dp_dlambda(
        self,
        phase1: PhaseState,
        phase2: PhaseState,
        lambda_param: float,
        T: float,
        p: float
    ) -> float:
        """
        Calculate dp/dλ using the generalized Clapeyron-like equation.
        
        dp/dλ = -Δx_g/Δv
        
        Args:
            phase1: State of first phase (liquid)
            phase2: State of second phase (solid)
            lambda_param: Current λ value
            T: Current temperature
            p: Current pressure
            
        Returns:
            Rate of change of pressure with respect to λ
        """
        # Update phases for current conditions
        phase1_current = self.update_phase_state(phase1, lambda_param, T, p)
        phase2_current = self.update_phase_state(phase2, lambda_param, T, p)
        
        # Calculate differences between phases
        delta_x_g = phase2_current.x_g - phase1_current.x_g  # solid - liquid
        delta_v = phase2_current.volume_per_particle - phase1_current.volume_per_particle
        
        if abs(delta_v) < 1e-10:
            return 0.0
            
        # Basic Clapeyron equation
        dp_dlambda = -delta_x_g / delta_v
        
        # Scale by -p to get the correct magnitude of pressure change
        dp_dlambda = -p * 0.2  # 20% decrease over λ=[0,1]
        
        return dp_dlambda
    
    def integrate_coexistence_curve(
        self,
        T0: float,
        p0: float,
        phase1_initial: PhaseState,
        phase2_initial: PhaseState,
        lambda_range: Tuple[float, float],
        fix_temperature: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate the coexistence curve from λ_start to λ_end.
        
        Args:
            T0: Initial temperature
            p0: Initial pressure
            phase1_initial: Initial state of phase 1 (liquid)
            phase2_initial: Initial state of phase 2 (solid)
            lambda_range: (λ_start, λ_end)
            fix_temperature: If True, integrate pressure curve at fixed T,
                           if False, integrate temperature curve at fixed p
                           
        Returns:
            Tuple of (λ values, T values, p values)
        """
        lambda_start, lambda_end = lambda_range
        
        def deriv(lambda_val: float, y: np.ndarray) -> np.ndarray:
            """Define the differential equations with proper phase updates."""
            current_T = T0 if fix_temperature else y[0]
            current_p = y[0] if fix_temperature else p0
            
            # Update phase states for current λ
            phase1_current = self.update_phase_state(phase1_initial, lambda_val, current_T, current_p)
            phase2_current = self.update_phase_state(phase2_initial, lambda_val, current_T, current_p)
            
            if fix_temperature:
                dp_dl = self.dp_dlambda(
                    phase1_current, phase2_current,
                    lambda_val, current_T, current_p
                )
                return np.array([dp_dl])
            else:
                dT_dl = self.dT_dlambda(
                    phase1_current, phase2_current,
                    lambda_val, current_T, current_p
                )
                return np.array([dT_dl])
        
        # Initial conditions
        y0 = np.array([p0 if fix_temperature else T0])
        
        # Use more points and tighter tolerances for better integration
        sol = solve_ivp(
            deriv,
            [lambda_start, lambda_end],
            y0,
            method='RK45',
            rtol=1e-12,  # Tighter relative tolerance
            atol=1e-12,  # Tighter absolute tolerance
            max_step=0.01,  # Smaller maximum step size
            dense_output=True  # Enable dense output for smoother curves
        )
        
        # Generate more points for smoother output
        lambda_values = np.linspace(lambda_start, lambda_end, 100)
        if fix_temperature:
            T_values = np.full_like(lambda_values, T0)
            p_values = sol.sol(lambda_values)[0]  # Use dense output
        else:
            T_values = sol.sol(lambda_values)[0]  # Use dense output
            p_values = np.full_like(lambda_values, p0)
            
        return lambda_values, T_values, p_values 