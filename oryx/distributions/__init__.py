"""
Oryx Distributions

Thin wrapper around distreqx.distributions to allow for easier imports and potential
future extensions.
"""

from distreqx.distributions import (
    AbstractDistribution,
    Bernoulli,
    Categorical,
    MultivariateNormalDiag,
    Normal,
)

__all__ = [
    "AbstractDistribution",
    "Bernoulli",
    "Categorical",
    "Normal",
    "MultivariateNormalDiag",
]
