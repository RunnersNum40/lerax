from __future__ import annotations

from distreqx import distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Integer

from .base_distribution import AbstractMaskableDistribution


class Categorical(AbstractMaskableDistribution[Integer[Array, ""]]):
    """
    Categorical distribution.

    Attributes:
        distribution: The underlying distreqx Categorical distribution.

    Args:
        logits: The log-odds of the distribution.
        probs: The probabilities of the distribution.
    """

    distribution: distributions.Categorical

    def __init__(
        self,
        logits: Float[ArrayLike, " dims"] | None = None,
        probs: Float[ArrayLike, " dims"] | None = None,
    ):
        logits = jnp.asarray(logits) if logits is not None else None
        probs = jnp.asarray(probs) if probs is not None else None

        self.distribution = distributions.Categorical(logits=logits, probs=probs)

    @property
    def logits(self) -> Float[Array, " dims"]:
        return self.distribution.logits

    @property
    def probs(self) -> Float[Array, " dims"]:
        return self.distribution.probs

    def mask(self, mask: Bool[Array, " dims"]) -> Categorical:
        masked_logits = jnp.where(mask, self.logits, -jnp.inf)
        return Categorical(logits=masked_logits)
