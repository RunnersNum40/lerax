from __future__ import annotations

from typing import Any, Self

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, Key, PyTree

from lerax.policy import AbstractPolicyState
from lerax.space import AbstractSpace

from .base_buffer import AbstractBuffer


class ReplayBuffer[StateType: AbstractPolicyState, ActType, ObsType, MaskType](
    AbstractBuffer
):
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
    action_masks: MaskType | None

    def __init__(
        self,
        size: int,
        observation_space: AbstractSpace[ObsType, Any],
        action_space: AbstractSpace[ActType, MaskType],
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
        self.action_masks = None

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
        action_mask: PyTree = None,
    ) -> Self:
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

        new_position = self.position + 1

        where_fns = [
            lambda rb: rb.position,
            lambda rb: rb.observations,
            lambda rb: rb.next_observations,
            lambda rb: rb.actions,
            lambda rb: rb.rewards,
            lambda rb: rb.dones,
            lambda rb: rb.timeouts,
        ]
        replacements = [
            new_position,
            observations,
            next_observations,
            actions,
            rewards,
            dones,
            timeouts,
        ]

        if self.states is not None:
            states = jax.tree.map(set_at_idx, self.states, state)
            next_states = jax.tree.map(set_at_idx, self.next_states, next_state)
            where_fns.append(lambda rb: rb.states)
            where_fns.append(lambda rb: rb.next_states)
            replacements.append(states)
            replacements.append(next_states)

        result = self
        for where_fn, replacement in zip(where_fns, replacements):
            result = eqx.tree_at(where_fn, result, replacement)
        return result

    def sample(
        self,
        batch_size: int,
        *,
        key: Key[Array, ""],
    ) -> Self:
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
