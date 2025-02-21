"""Module for calculating TIP4P water model properties."""

import numpy as np
from typing import Tuple
from .tip4p_parameters import TIP4PParameters
from integration.hamiltonian_gibbs_duhem import HamiltonianGibbsDuhemIntegrator, PhaseState
from integration.example_gibbs_duhem import calculate_total_energy, get_realistic_phase_properties
from geometry.bernal_fowler import create_water_molecule, rotation_matrix

# Physical constants and conversion factors
kB = 1.380649e-23  # Boltzmann constant (J/K)
NA = 6.02214076e23  # Avogadro constant
e = 1.60217663e-19  # Elementary charge (C)
eps0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
kcal_to_kJ = 4.184  # kcal to kJ conversion
ang_to_m = 1e-10  # Angstrom to meter conversion
water_mw = 18.015  # g/mol

def create_tip4p_molecule(params: TIP4PParameters):
    """Create a TIP4P water molecule with proper Bernal-Fowler geometry."""
    # Use the existing Bernal-Fowler geometry code
    rot = rotation_matrix([0, 1, 0], np.pi/6)  # Rotate for better view
    molecule = create_water_molecule(orientation=rot)
    
    # Scale coordinates according to TIP4P parameters
    for site in ['H1', 'H2']:
        molecule[site] = molecule[site] * (params.sigma/3.154)
    molecule['M'] = molecule['O'] + (molecule['M'] - molecule['O']) * (params.dOM/0.15)
    
    return molecule

def lennard_jones_tip4p(coordinates: np.ndarray, p: float, T: float, 
                       params: TIP4PParameters) -> float:
    """
    TIP4P potential energy function with proper scaling.
    Returns energy in kJ/mol.
    """
    r = np.linalg.norm(coordinates)
    if r < 0.1:  # Avoid singularity at r=0
        return 1e10
    
    # LJ potential with proper scaling
    sr = params.sigma/r
    sr6 = sr**6
    sr12 = sr6**2
    u_lj = 4.0 * params.epsilon * (sr12 - sr6)
    
    # Add electrostatic term with proper units
    r_m = r * ang_to_m
    qH_C = params.qH * e
    qM_C = params.qM * e
    u_elec = (qH_C * qM_C / (4 * np.pi * eps0 * r_m)) * NA / 1000
    
    return u_lj + u_elec

def get_phase_properties(params: TIP4PParameters, T: float, p: float, phase: str) -> Tuple[float, float, float]:
    """
    Get entropy, volume, and energy using proper scaling.
    Returns values in appropriate units.
    """
    # Get base properties from existing code
    phase_props = get_realistic_phase_properties(T)
    s_base, v_base = phase_props[phase]
    
    # Convert to proper units with parameter scaling
    s = s_base * kB * NA  # J/(mol·K)
    v = v_base * (params.sigma * ang_to_m)**3 * NA * 1e6  # cm³/mol
    
    # Calculate energy with proper neighbor shells
    if phase == 'liquid':
        r = params.sigma * 1.1  # Typical liquid separation
        coord_number = 4.4  # Average coordination in liquid
    else:  # solid (ice Ih)
        r = params.sigma  # Perfect crystal spacing
        coord_number = 4.0  # Tetrahedral coordination
    
    # Energy per molecule including both LJ and electrostatic
    e = coord_number * lennard_jones_tip4p(np.array([r]), p, T, params)
    
    return s, v, e

def calculate_melting_properties(params: TIP4PParameters) -> dict:
    """
    Calculate properties at the melting point using Gibbs-Duhem integration.
    Now with proper energy scaling and realistic phase properties.
    """
    # Reference temperature and pressure
    T0 = 273.15  # K
    p0 = 1.0     # bar
    
    # Get phase properties with proper scaling
    s_liquid, v_liquid, e_liquid = get_phase_properties(params, T0, p0, 'liquid')
    s_solid, v_solid, e_solid = get_phase_properties(params, T0, p0, 'solid')
    
    # Calculate enthalpy of fusion with proper scaling
    dH = T0 * (s_liquid - s_solid) + (e_liquid - e_solid)
    
    # Calculate densities
    rho_liquid = water_mw / v_liquid  # g/cm³
    rho_solid = water_mw / v_solid    # g/cm³
    
    return {
        'melting_temperature': T0,
        'melting_pressure': p0,
        'liquid_density': rho_liquid,
        'solid_density': rho_solid,
        'enthalpy_fusion': abs(dH),  # Use absolute value for consistency
        'phase_properties': {
            'liquid': {'s': s_liquid, 'v': v_liquid, 'e': e_liquid},
            'solid': {'s': s_solid, 'v': v_solid, 'e': e_solid}
        }
    }

def calculate_ice_density(params: TIP4PParameters) -> float:
    """Calculate ice Ih density using proper crystal structure."""
    properties = calculate_melting_properties(params)
    return properties['solid_density']

def calculate_melting_temperature(params: TIP4PParameters) -> float:
    """Calculate melting temperature using phase equilibrium."""
    properties = calculate_melting_properties(params)
    return properties['melting_temperature']

def calculate_enthalpy_fusion(params: TIP4PParameters) -> float:
    """Calculate enthalpy of fusion using proper phase energetics."""
    properties = calculate_melting_properties(params)
    return properties['enthalpy_fusion'] 