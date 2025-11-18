from __future__ import annotations

from dataclasses import replace
from functools import partial

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, Key, PyTree

from lerax.policy import AbstractPolicyState
from lerax.space import AbstractSpace

from .base_buffer import AbstractBuffer


class ReplayBuffer[StateType: AbstractPolicyState, ActType, ObsType](AbstractBuffer):
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
        size_arr = jnp.asarray(size, dtype=int)
        if size_arr.ndim != 0:
            raise ValueError("ReplayBuffer size must be a scalar.")

        self.size = size_arr
        self.position = jnp.array(0, dtype=int)

        tile = partial(jnp.tile, reps=(self.size,))
        self.observations = jax.tree.map(tile, observation_space.canonical())
        self.next_observations = jax.tree.map(tile, observation_space.canonical())
        self.actions = jax.tree.map(tile, action_space.canonical())
        self.rewards = jnp.zeros((self.size,), dtype=float)
        self.dones = jnp.zeros((self.size,), dtype=bool)
        self.timeouts = jnp.zeros((self.size,), dtype=bool)
        self.states = jax.tree.map(tile, state)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rewards.shape

    @property
    def current_size(self) -> Integer[Array, ""]:
        return jnp.minimum(self.position, self.size)

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

        def set_at_idx(leaf, new_value):
            return leaf.at[idx].set(new_value)

        observations = jax.tree.map(set_at_idx, self.observations, observation)
        next_observations = jax.tree.map(
            set_at_idx, self.next_observations, next_observation
        )
        actions = jax.tree.map(set_at_idx, self.actions, action)
        states = jax.tree.map(set_at_idx, self.states, state)

        rewards = self.rewards.at[idx].set(reward)
        dones = self.dones.at[idx].set(done)
        timeouts = self.timeouts.at[idx].set(timeout)

        new_position = self.position + 1

        return replace(
            self,
            position=new_position,
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
        current_size = self.current_size
        current_size = eqx.error_if(
            current_size,
            current_size < batch_size,
            "Cannot sample more elements than are currently stored in the buffer.",
        )

        batch_indices = jr.choice(key, current_size, shape=(batch_size,), replace=False)

        def take_at_indices(leaf):
            return jnp.take(leaf, batch_indices, axis=0)

        observations = jax.tree.map(take_at_indices, self.observations)
        next_observations = jax.tree.map(take_at_indices, self.next_observations)
        actions = jax.tree.map(take_at_indices, self.actions)
        states = jax.tree.map(take_at_indices, self.states)

        rewards = take_at_indices(self.rewards)
        dones = take_at_indices(self.dones)
        timeouts = take_at_indices(self.timeouts)

        batch_size_arr = jnp.array(batch_size, dtype=int)

        return replace(
            self,
            size=batch_size_arr,
            position=batch_size_arr,
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            states=states,
        )
