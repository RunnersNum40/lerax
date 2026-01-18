from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Literal

import equinox as eqx
from jax import nn as jnn
from jax import numpy as jnp
from jaxtyping import Array, Float, Key, ScalarLike


class AbstractNCDETerm(eqx.Module):
    """Base class for a neural ODE vector field term."""

    @abstractmethod
    def __call__(
        self, t: ScalarLike, z: Float[Array, " data_size"], args
    ) -> Float[Array, " data_size input_size"]:
        """Vector field value."""


class TensorMLP(eqx.Module):
    """
    Modification of the MLP class from Equinox to handle tensors as input and output.
    """

    in_shape: tuple[int, ...] | Literal["scalar"]
    out_shape: tuple[int, ...] | Literal["scalar"]
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_shape: tuple[int, ...] | Literal["scalar"],
        out_shape: tuple[int, ...] | Literal["scalar"],
        width_size: int,
        depth: int,
        activation: Callable[[Array], Array] = jnn.relu,
        final_activation: Callable[[Array], Array] = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: jnp.dtype | None = None,
        *,
        key: Key,
    ) -> None:
        self.in_shape = in_shape
        self.out_shape = out_shape

        if in_shape == "scalar":
            in_size = "scalar"
        else:
            in_size = int(jnp.asarray(in_shape).prod())

        if out_shape == "scalar":
            out_size = "scalar"
        else:
            out_size = int(jnp.asarray(out_shape).prod())

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            dtype=dtype,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        if self.in_shape == "scalar":
            x = self.mlp(x)
        else:
            x = jnp.ravel(x)
            x = self.mlp(x)

        if self.out_shape != "scalar":
            x = jnp.reshape(x, self.out_shape)

        return x


class MLPNCDETerm(AbstractNCDETerm):
    add_time: bool
    tensor_mlp: TensorMLP

    def __init__(
        self,
        input_size: int,
        state_size: int,
        width_size: int,
        depth: int,
        activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = jnn.softplus,
        final_activation: Callable[
            [Float[Array, " width"]], Float[Array, " width"]
        ] = jnn.tanh,
        *,
        key: Key,
        add_time: bool = True,
    ):
        self.add_time = add_time
        self.tensor_mlp = TensorMLP(
            in_shape=(state_size + int(add_time),),
            out_shape=(state_size, input_size),
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=key,
        )

    def __call__(
        self, t: ScalarLike, z: Float[Array, " state_size"], args
    ) -> Float[Array, " state_size input_size"]:
        if self.add_time:
            return self.tensor_mlp(
                jnp.concatenate([z, jnp.expand_dims(t, axis=-1)], axis=-1)
            )
        else:
            return self.tensor_mlp(z)
