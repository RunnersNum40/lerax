"""
Lerax spaces
"""

from .base_space import AbstractSpace
from .box import Box
from .dict import Dict
from .discrete import Discrete
from .multi_binary import MultiBinary
from .multi_discrete import MultiDiscrete
from .tuple import Tuple

__all__ = [
    "AbstractSpace",
    "Box",
    "Dict",
    "Discrete",
    "MultiBinary",
    "MultiDiscrete",
    "Tuple",
]
