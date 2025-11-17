from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, Key, PyTree

from lerax.policy import AbstractPolicyState
from lerax.space import AbstractSpace

from .base_buffer import AbstractBuffer


class ReplayBuffer[StateType: AbstractPolicyState, ActType, ObsType](AbstractBuffer):
    """ReplayBuffer used by off-policy algorithms."""

    size: Integer[Array, ""]
    position: Integer[Array, ""]

    observations: PyTree[ObsType]
    next_observations: PyTree[ObsType]
    actions: PyTree[ActType]
    rewards: Float[Array, " capacity"]
    dones: Bool[Array, " capacity"]
    timeouts: Bool[Array, " capacity"]
    states: StateType

    def __init__(
        self,
        size: Integer[ArrayLike, ""],
        observation_space: AbstractSpace[ObsType],
        action_space: AbstractSpace[ActType],
        state: StateType,
    ):
        self.size = jnp.array(size, dtype=int)
        self.position = jnp.array(0, dtype=int)

        tile = partial(jnp.tile, reps=(size,))
        self.observations = jax.tree.map(tile, observation_space.canonical())
        self.next_observations = jax.tree.map(tile, observation_space.canonical())
        self.actions = jax.tree.map(tile, action_space.canonical())
        self.rewards = jnp.zeros((size,), dtype=float)
        self.dones = jnp.zeros((size,), dtype=bool)
        self.timeouts = jnp.zeros((size,), dtype=bool)
        self.states = jax.tree.map(tile, state)

    def add(
        self,
        observation: ObsType,
        next_observation: ObsType,
        action: ActType,
        reward: Float[ArrayLike, ""],
        done: Bool[ArrayLike, ""],
        timeout: Bool[ArrayLike, ""],
        state: StateType,
    ) -> ReplayBuffer[StateType, ActType, ObsType]:
        reward = jnp.asarray(reward, dtype=float)
        done = jnp.asarray(done, dtype=bool)
        timeout = jnp.asarray(timeout, dtype=bool)

        idx = self.position % self.size

        observations = jax.tree.map(
            lambda leaf, new: leaf.at[idx].set(new), self.observations, observation
        )
        next_observations = jax.tree.map(
            lambda leaf, new: leaf.at[idx].set(new),
            self.next_observations,
            next_observation,
        )
        actions = jax.tree.map(
            lambda leaf, new: leaf.at[idx].set(new), self.actions, action
        )
        rewards = self.rewards.at[idx].set(reward)
        dones = self.dones.at[idx].set(done)
        timeouts = self.timeouts.at[idx].set(timeout)
        states = jax.tree.map(
            lambda leaf, new: leaf.at[idx].set(new), self.states, state
        )

        return replace(
            self,
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            states=states,
        )

    def sample(
        self, batch_size: int, *, key: Key
    ) -> ReplayBuffer[StateType, ActType, ObsType]:
        batch_indices = jr.choice(
            key, self.position, shape=(batch_size,), replace=False
        )
        return jax.tree.map(partial(jnp.take, indices=batch_indices, axis=0), self)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rewards.shape
