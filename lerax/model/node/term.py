from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jax import nn as jnn
from jax import numpy as jnp
from jaxtyping import Array, Float, Key, ScalarLike


class AbstractNODETerm(eqx.Module):
    """Base class for a neural ODE vector field term."""

    @abstractmethod
    def __call__(
        self, t: ScalarLike, z: Float[Array, " data_size"], args
    ) -> Float[Array, " data_size"]:
        """Vector field value."""


class MLPNODETerm(AbstractNODETerm):
    add_time: bool
    mlp: eqx.nn.MLP

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        activation: Callable[[Array], Float[Array, " data_size"]] = jnn.softplus,
        final_activation: Callable[[Array], Float[Array, " data_size"]] = jnn.tanh,
        *,
        key: Key,
        add_time: bool = True,
    ):
        self.add_time = add_time
        self.mlp = eqx.nn.MLP(
            in_size=data_size + int(add_time),
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=key,
        )

    def __call__(
        self, t: ScalarLike, z: Float[Array, " data_size"], args
    ) -> Float[Array, " data_size"]:
        if self.add_time:
            return self.mlp(jnp.concatenate([z, jnp.expand_dims(t, axis=-1)], axis=-1))
        else:
            return self.mlp(z)
