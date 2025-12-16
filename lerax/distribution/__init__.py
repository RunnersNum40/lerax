"""
Lerax Distributions

Wrapper around distreqx.distributions to allow for easier imports, extended typing, and
future expansion.
"""

from .base_distribution import (
    AbstractDistreqxWrapper,
    AbstractDistribution,
    AbstractMaskableDistribution,
    AbstractTransformedDistribution,
)
from .bernoulli import Bernoulli
from .categorical import Categorical
from .multi_categorical import MultiCategorical
from .multivariate_normal import MultivariateNormalDiag
from .normal import Normal
from .squashed_multivariate_normal import SquashedMultivariateNormalDiag
from .squashed_normal import SquashedNormal

__all__ = [
    "AbstractDistribution",
    "AbstractDistreqxWrapper",
    "AbstractMaskableDistribution",
    "AbstractTransformedDistribution",
    "Bernoulli",
    "Categorical",
    "Normal",
    "MultivariateNormalDiag",
    "SquashedMultivariateNormalDiag",
    "SquashedNormal",
    "MultiCategorical",
]
