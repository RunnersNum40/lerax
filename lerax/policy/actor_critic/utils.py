from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int, Key

from lerax.distribution import (
    AbstractDistribution,
    Bernoulli,
    Categorical,
    SquashedMultivariateNormalDiag,
    SquashedNormal,
)
from lerax.model import AbstractModel
from lerax.space import AbstractSpace, Box, Discrete, MultiBinary


class AbstractActionLayer[ActType](AbstractModel):
    """Layer that produces action distributions given inputs."""

    mapping: eqx.AbstractVar[eqx.nn.Linear]

    @abstractmethod
    def __call__(
        self,
        inputs: Float[Array, " latent_dim"],
    ) -> AbstractDistribution[ActType]:
        """Produce an action distribution given inputs."""


class BoxActionLayer(AbstractActionLayer[Float[Array, " action_dim"]]):

    scalar: bool
    mapping: eqx.nn.Linear
    log_std: Float[Array, " action_dim"]

    def __init__(
        self, latent_dim: int, action_space: Box, *, key: Key, log_std_init: float = 0.0
    ):
        if action_space.shape:
            self.scalar = False
            self.mapping = eqx.nn.Linear(
                latent_dim, int(action_space.flat_size), key=key
            )
            self.log_std = jnp.full(int(action_space.flat_size), log_std_init)
        else:
            self.scalar = True
            self.mapping = eqx.nn.Linear(latent_dim, "scalar", key=key)
            self.log_std = jnp.array(log_std_init)

    def __call__(
        self, inputs: Float[Array, " latent_dim"]
    ) -> SquashedNormal | SquashedMultivariateNormalDiag:
        if self.scalar:
            return SquashedNormal(loc=self.mapping(inputs), scale=jnp.exp(self.log_std))
        else:
            return SquashedMultivariateNormalDiag(
                loc=self.mapping(inputs), scale_diag=jnp.exp(self.log_std)
            )


class CategoricalActionLayer(AbstractActionLayer[Int[Array, ""]]):

    mapping: eqx.nn.Linear

    def __init__(self, latent_dim: int, action_space: Discrete, *, key: Key):
        self.mapping = eqx.nn.Linear(latent_dim, int(action_space.n), key=key)

    def __call__(self, inputs: Float[Array, " latent_dim"]) -> Categorical:
        return Categorical(logits=self.mapping(inputs))


class BernoulliActionLayer(AbstractActionLayer[Int[Array, ""]]):

    mapping: eqx.nn.Linear

    def __init__(self, latent_dim: int, action_space: MultiBinary, *, key: Key):
        self.mapping = eqx.nn.Linear(latent_dim, int(action_space.n), key=key)

    def __call__(self, inputs: Float[Array, " latent_dim"]) -> Bernoulli:
        return Bernoulli(logits=self.mapping(inputs))


def make_action_layer(
    latent_dim: int, action_space: AbstractSpace, *, key: Key, log_std_init: float = 0.0
) -> AbstractActionLayer:
    """Create an action layer based on the action space."""

    if isinstance(action_space, Box):
        return BoxActionLayer(
            latent_dim, action_space, key=key, log_std_init=log_std_init
        )
    elif isinstance(action_space, Discrete):
        return CategoricalActionLayer(latent_dim, action_space, key=key)
    elif isinstance(action_space, MultiBinary):
        return BernoulliActionLayer(latent_dim, action_space, key=key)
    else:
        raise NotImplementedError(f"Action space {type(action_space)} not supported.")


class ActionHead[ActType](AbstractModel):
    mlp: eqx.nn.MLP | None
    action_layer: AbstractActionLayer[ActType]

    def __init__(
        self,
        action_space: AbstractSpace,
        latent_dim: int,
        width_size: int,
        depth: int,
        *,
        key: Key,
        log_std_init: float = 0.0,
    ):
        mlp_key, action_key = jr.split(key, 2)

        depth = max(0, depth - 1)
        if depth < 0:
            self.mlp = None
        else:
            self.mlp = eqx.nn.MLP(
                in_size=latent_dim,
                out_size=latent_dim,
                width_size=width_size,
                depth=depth,
                key=mlp_key,
            )
        self.action_layer = make_action_layer(
            latent_dim, action_space, key=action_key, log_std_init=log_std_init
        )

    def __call__(
        self, features: Float[Array, " latent_dim"]
    ) -> AbstractDistribution[ActType]:
        if self.mlp is not None:
            features = self.mlp(features)
        return self.action_layer(features)
