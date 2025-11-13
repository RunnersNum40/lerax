from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Bool,
    Float,
    PyTree,
)

from lerax.policy import AbstractPolicyState

from .base_buffer import AbstractBuffer


class ReplayBuffer[StateType: AbstractPolicyState, ActType, ObsType](AbstractBuffer):
    """
    ReplayBuffer used by off-policy algorithms.

    The buffer is implemented as a fixed-size circular buffer. New transitions
    overwrite the oldest ones once the capacity is reached.

    Observations and actions are stored as PyTrees of arrays, with a leading
    buffer dimension. This supports arbitrary nested structures as long as
    they are JAX arrays.
    """

    observations: PyTree[ObsType]
    next_observations: PyTree[ObsType]
    actions: PyTree[ActType]
    rewards: Float[Array, " capacity"]
    dones: Bool[Array, " capacity"]
    timeouts: Bool[Array, " capacity"]
    states: StateType

    def __init__(
        self,
        observations: PyTree[ObsType],
        next_observations: PyTree[ObsType],
        actions: PyTree[ActType],
        rewards: Float[ArrayLike, " capacity"],
        dones: Bool[ArrayLike, " capacity"],
        timeouts: Bool[ArrayLike, " capacity"],
        states: StateType,
    ):
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = jnp.asarray(rewards, dtype=float)
        self.dones = jnp.asarray(dones, dtype=bool)
        self.timeouts = jnp.asarray(timeouts, dtype=bool)
        self.states = states

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rewards.shape
