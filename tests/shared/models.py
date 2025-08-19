from __future__ import annotations

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array

from oryx.model.base_model import AbstractModel, AbstractStatefulModel


class Doubler(AbstractModel[[Array], Array]):
    def __call__(self, x: Array) -> Array:
        return 2.0 * x


class StatefulDoubler(AbstractStatefulModel[[Array], Array]):
    state_index: eqx.nn.StateIndex[Array]

    def __init__(self):
        self.state_index = eqx.nn.StateIndex(jnp.array(0.0))

    def __call__(self, state, x):
        c = state.get(self.state_index)
        state = state.set(self.state_index, c + 1.0)
        return state, 2.0 * x

    def reset(self, state):
        return state.set(self.state_index, jnp.array(0.0))
