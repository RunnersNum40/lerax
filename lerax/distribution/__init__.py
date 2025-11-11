"""
Lerax Distributions

Wrapper around distreqx.distributions to allow for easier imports, extended typing, and
future expansion.
"""

from .base_distribution import (
    AbstractDistribution,
    AbstractMaskableDistribution,
    AbstractTransformedDistribution,
)
from .distributions import (
    Bernoulli,
    Categorical,
    MultivariateNormalDiag,
    Normal,
    SquashedMultivariateNormalDiag,
    SquashedNormal,
)

__all__ = [
    "AbstractDistribution",
    "AbstractMaskableDistribution",
    "AbstractTransformedDistribution",
    "Bernoulli",
    "Categorical",
    "Normal",
    "MultivariateNormalDiag",
    "SquashedMultivariateNormalDiag",
    "SquashedNormal",
]
