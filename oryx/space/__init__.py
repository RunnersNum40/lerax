"""
Oryx spaces
"""

from .base_space import (
    AbstractSpace,
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Tuple,
)
from .utils import flat_dim, flatten

__all__ = [
    "AbstractSpace",
    "Box",
    "Dict",
    "Discrete",
    "MultiBinary",
    "MultiDiscrete",
    "Tuple",
    "flat_dim",
    "flatten",
]
