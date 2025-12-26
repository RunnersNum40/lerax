from __future__ import annotations

from abc import abstractmethod
from typing import cast

import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Key

from lerax.distribution import (
    AbstractDistribution,
    AbstractMaskableDistribution,
    Bernoulli,
    Categorical,
    MultiCategorical,
    MultivariateNormalDiag,
    Normal,
)
from lerax.space import AbstractSpace, Box, Discrete, MultiBinary, MultiDiscrete

from .base_model import AbstractModel


class AbstractActionDistribution[ActType, MaskType](AbstractModel):
    """Layer that produces action distributions given inputs."""

    mapping: eqx.AbstractVar[eqx.nn.Linear]

    @abstractmethod
    def __call__(
        self,
        inputs: Float[Array, " latent_dim"],
    ) -> (
        AbstractDistribution[ActType] | AbstractMaskableDistribution[ActType, MaskType]
    ):
        """Produce an action distribution given inputs."""


class BoxAction(AbstractActionDistribution[Float[Array, " action_dim"], None]):

    scalar: bool
    mapping: eqx.nn.Linear
    log_std: Float[Array, " *action_dim"]

    def __init__(
        self,
        latent_dim: int,
        action_space: Box,
        *,
        key: Key,
        log_std_init: Float[ArrayLike, ""] = jnp.array(0.0),
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
    ) -> Normal | MultivariateNormalDiag:
        if self.scalar:
            return Normal(
                loc=self.mapping(inputs),
                scale=jnp.exp(self.log_std),
            )
        else:
            return MultivariateNormalDiag(
                loc=self.mapping(inputs),
                scale_diag=jnp.exp(self.log_std),
            )


class DiscreteAction(AbstractActionDistribution[Int[Array, ""], Bool[Array, " n"]]):

    mapping: eqx.nn.Linear

    def __init__(self, latent_dim: int, action_space: Discrete, *, key: Key):
        self.mapping = eqx.nn.Linear(latent_dim, int(action_space.n), key=key)

    def __call__(self, inputs: Float[Array, " latent_dim"]) -> Categorical:
        return Categorical(logits=self.mapping(inputs))


class MultiBinaryAction(AbstractActionDistribution[Int[Array, ""], None]):

    mapping: eqx.nn.Linear
    shape: tuple[int, ...]

    def __init__(self, latent_dim: int, action_space: MultiBinary, *, key: Key):
        self.mapping = eqx.nn.Linear(latent_dim, action_space.flat_size, key=key)
        self.shape = action_space.shape

    def __call__(self, inputs: Float[Array, " latent_dim"]) -> Bernoulli:
        return Bernoulli(logits=self.mapping(inputs).reshape(self.shape))


class MultiDiscreteAction(AbstractActionDistribution[Int[Array, ""], None]):

    ns: tuple[int, ...]
    mappings: eqx.nn.Linear
    shape: tuple[int, ...]

    def __init__(self, latent_dim: int, action_space: MultiDiscrete, *, key: Key):
        self.ns = action_space.nvec
        self.mappings = eqx.nn.Linear(latent_dim, sum(action_space.nvec), key=key)
        self.shape = action_space.shape

    def __call__(self, inputs: Float[Array, " latent_dim"]) -> MultiCategorical:
        return MultiCategorical(self.mappings(inputs), action_dims=self.ns)


def make_action_layer[ActType, MaskType](
    latent_dim: int,
    action_space: AbstractSpace[ActType, MaskType],
    *,
    key: Key,
    log_std_init: float = 0.0,
) -> AbstractActionDistribution[ActType, MaskType]:
    """Create an action layer based on the action space."""

    if isinstance(action_space, Box):
        return cast(
            AbstractActionDistribution[ActType, MaskType],
            BoxAction(latent_dim, action_space, key=key, log_std_init=log_std_init),
        )
    elif isinstance(action_space, Discrete):
        return cast(
            AbstractActionDistribution[ActType, MaskType],
            DiscreteAction(latent_dim, action_space, key=key),
        )
    elif isinstance(action_space, MultiBinary):
        return cast(
            AbstractActionDistribution[ActType, MaskType],
            MultiBinaryAction(latent_dim, action_space, key=key),
        )
    elif isinstance(action_space, MultiDiscrete):
        return cast(
            AbstractActionDistribution[ActType, MaskType],
            MultiDiscreteAction(latent_dim, action_space, key=key),
        )
    else:
        raise NotImplementedError(f"Action space {type(action_space)} not supported.")


class ActionLayer[ActType, MaskType](AbstractModel):
    """
    Model that produces action distributions from features.

    An optional MLP processes the features before passing them to the action distribution layer.
    If depth is set to 1, no MLP is used and the features are affine-mapped directly to the action parameters.

    Attributes:
        mlp: Optional MLP to process features before action distribution.
        action_dist: Action distribution layer.

    Args:
        action_space: The action space defining the type of actions.
        latent_dim: Dimension of the input features.
        width_size: Width of the hidden layers in the MLP.
        depth: Depth of the MLP. If set to 1, no MLP is
        log_std_init: Initial log standard deviation for continuous action spaces.
        key: JAX PRNG key for parameter initialization.
    """

    mlp: eqx.nn.MLP | None
    action_dist: AbstractActionDistribution[ActType, MaskType]

    def __init__(
        self,
        action_space: AbstractSpace[ActType, MaskType],
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
        self.action_dist = make_action_layer(
            latent_dim, action_space, key=action_key, log_std_init=log_std_init
        )

    def __call__(
        self, features: Float[Array, " latent_dim"], action_mask: MaskType | None = None
    ) -> AbstractDistribution[ActType]:
        if self.mlp is not None:
            features = self.mlp(features)

        dist = self.action_dist(features)

        if isinstance(dist, AbstractMaskableDistribution) and action_mask is not None:
            dist = dist.mask(action_mask)

        return dist
