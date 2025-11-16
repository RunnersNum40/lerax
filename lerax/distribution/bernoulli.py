from __future__ import annotations

from distreqx import distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

from .base_distribution import AbstractMaskableDistribution


class Bernoulli(AbstractMaskableDistribution[Bool[Array, " dims"]]):

    distribution: distributions.Bernoulli

    def __init__(
        self,
        logits: Float[ArrayLike, " dims"] | None = None,
        probs: Float[ArrayLike, " dims"] | None = None,
    ):
        logits = jnp.asarray(logits) if logits is not None else None
        probs = jnp.asarray(probs) if probs is not None else None
        self.distribution = distributions.Bernoulli(logits=logits, probs=probs)

    @property
    def logits(self) -> Float[Array, " dims"]:
        return self.distribution.logits

    @property
    def probs(self) -> Float[Array, " dims"]:
        return self.distribution.probs

    def mask(self, mask: Bool[Array, " dims"]) -> Bernoulli:
        masked_logits = jnp.where(mask, self.logits, -jnp.inf)
        return Bernoulli(logits=masked_logits)
