"""
Oryx spaces

AbstractSpace, Box, Dict, Discrete, MultiDiscrete, Tuple
"""

from .base_space import (
    AbstractSpace,
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    OneOf,
    Tuple,
)
from .utils import flatten

__all__ = [
    "AbstractSpace",
    "Box",
    "Dict",
    "Discrete",
    "MultiBinary",
    "MultiDiscrete",
    "OneOf",
    "Tuple",
    "flatten",
]
