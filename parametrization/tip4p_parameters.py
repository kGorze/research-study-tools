"""Module containing the TIP4PParameters dataclass."""

from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Tuple, Callable

@dataclass
class TIP4PParameters:
    """Class to hold TIP4P model parameters."""
    epsilon: float  # LJ well depth (kJ/mol)
    sigma: float   # LJ size parameter (Å)
    qH: float     # H-site charge (e)
    dOM: float = 0.150  # O-M distance (Å)
    
    @property
    def qM(self) -> float:
        """M-site charge determined by H-site charge."""
        return -2 * self.qH
    
    def as_array(self) -> np.ndarray:
        """Convert parameters to numpy array."""
        return np.array([self.epsilon, self.sigma, self.qH, self.dOM])
    
    @classmethod
    def from_array(cls, params: np.ndarray) -> 'TIP4PParameters':
        """Create TIP4PParameters from numpy array."""
        return cls(epsilon=params[0], sigma=params[1], 
                  qH=params[2], dOM=params[3]) 