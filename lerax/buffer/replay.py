from __future__ import annotations

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, Key, PyTree

from lerax.policy import AbstractPolicyState
from lerax.space import AbstractSpace

from .base_buffer import AbstractBuffer


class ReplayBuffer[StateType: AbstractPolicyState, ActType, ObsType](AbstractBuffer):
    size: int = eqx.field(static=True)
    position: Integer[Array, ""]

    observations: PyTree[ObsType]
    next_observations: PyTree[ObsType]
    actions: PyTree[ActType]
    rewards: Float[Array, " capacity"]
    dones: Bool[Array, " capacity"]
    timeouts: Bool[Array, " capacity"]
    states: StateType
    next_states: StateType

    def __init__(
        self,
        size: int,
        observation_space: AbstractSpace[ObsType],
        action_space: AbstractSpace[ActType],
        state: StateType,
    ):
        self.size = size
        self.position = jnp.array(0, dtype=int)

        def init_leaf(example):
            arr = jnp.asarray(example)
            return jnp.broadcast_to(arr, (self.size,) + arr.shape)

        self.observations = jax.tree.map(init_leaf, observation_space.canonical())
        self.next_observations = jax.tree.map(init_leaf, observation_space.canonical())
        self.actions = jax.tree.map(init_leaf, action_space.canonical())

        self.rewards = jnp.zeros((self.size,), dtype=float)
        self.dones = jnp.zeros((self.size,), dtype=bool)
        self.timeouts = jnp.zeros((self.size,), dtype=bool)

        self.states = jax.tree.map(init_leaf, state)
        self.next_states = jax.tree.map(init_leaf, state)

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
        next_state: StateType,
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
        rewards = self.rewards.at[idx].set(reward)
        dones = self.dones.at[idx].set(done)
        timeouts = self.timeouts.at[idx].set(timeout)
        states = jax.tree.map(set_at_idx, self.states, state)
        next_states = jax.tree.map(set_at_idx, self.next_states, next_state)

        new_position = self.position + 1

        return eqx.tree_at(
            lambda rb: (
                rb.position,
                rb.observations,
                rb.next_observations,
                rb.actions,
                rb.rewards,
                rb.dones,
                rb.timeouts,
                rb.states,
                rb.next_states,
            ),
            self,
            (
                new_position,
                observations,
                next_observations,
                actions,
                rewards,
                dones,
                timeouts,
                states,
                next_states,
            ),
        )

    def batches(
        self,
        batch_size: int,
        *,
        key: Key | None = None,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> ReplayBuffer[StateType, ActType, ObsType]:
        _ = eqx.error_if(
            self.current_size,
            self.current_size < self.size,
            "ReplayBuffer.batches assumes the buffer is full.",
        )

        flat_self = self.flatten_axes(batch_axes)

        total = flat_self.rewards.shape[0]
        indices = jnp.arange(total) if key is None else jr.permutation(key, total)

        if total % batch_size != 0:
            total_trim = total - (total % batch_size)
            indices = indices[:total_trim]

        indices = indices.reshape(-1, batch_size)

        def take_batch(x):
            if not isinstance(x, jnp.ndarray) or x.ndim == 0:
                return x
            return jnp.take(x, indices, axis=0)

        return jax.tree.map(take_batch, flat_self)

    def sample(
        self,
        batch_size: int,
        *,
        key: Key,
    ) -> ReplayBuffer[StateType, ActType, ObsType]:
        flat_self = self.flatten_axes(None)
        total = flat_self.rewards.shape[0]

        current_size = self.current_size

        if current_size.ndim == 0:
            valid_mask = jnp.arange(self.size) < current_size
        else:
            valid_mask = (jnp.arange(self.size) < current_size[..., None]).reshape(-1)

        probs = valid_mask.astype(float) / jnp.sum(valid_mask)
        batch_indices = jr.choice(
            key,
            total,
            shape=(batch_size,),
            replace=False,
            p=probs,
        )

        def take_sample(x):
            if not isinstance(x, jnp.ndarray) or x.ndim == 0:
                return x
            return jnp.take(x, batch_indices, axis=0)

        batch = jax.tree.map(take_sample, flat_self)

        return batch
