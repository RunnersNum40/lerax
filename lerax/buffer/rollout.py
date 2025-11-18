from __future__ import annotations

import dataclasses

import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike, Bool, Float, Key, PyTree

from lerax.policy import AbstractPolicyState

from .base_buffer import AbstractBuffer


class RolloutBuffer[StateType: AbstractPolicyState, ActType, ObsType](AbstractBuffer):
    observations: PyTree[ObsType]
    actions: PyTree[ActType]
    rewards: Float[Array, " *size"]
    dones: Bool[Array, " *size"]
    log_probs: Float[Array, " *size"]
    values: Float[Array, " *size"]
    returns: Float[Array, " *size"]
    advantages: Float[Array, " *size"]
    states: StateType

    def __init__(
        self,
        observations: PyTree[ObsType],
        actions: PyTree[ActType],
        rewards: Float[ArrayLike, " *size"],
        dones: Bool[ArrayLike, " *size"],
        log_probs: Float[ArrayLike, " *size"],
        values: Float[ArrayLike, " *size"],
        states: StateType,
        returns: Float[ArrayLike, " *size"] | None = None,
        advantages: Float[ArrayLike, " *size"] | None = None,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = jnp.asarray(rewards, dtype=float)
        self.dones = jnp.asarray(dones, dtype=bool)
        self.log_probs = jnp.asarray(log_probs, dtype=float)
        self.values = jnp.asarray(values, dtype=float)
        self.states = states
        self.returns = (
            jnp.asarray(returns, dtype=float)
            if returns is not None
            else jnp.full_like(values, jnp.nan, dtype=float)
        )
        self.advantages = (
            jnp.asarray(advantages, dtype=float)
            if advantages is not None
            else jnp.full_like(values, jnp.nan, dtype=float)
        )

    def compute_returns_and_advantages(
        self,
        last_value: Float[ArrayLike, ""],
        gae_lambda: Float[ArrayLike, ""],
        gamma: Float[ArrayLike, ""],
    ) -> RolloutBuffer[StateType, ActType, ObsType]:
        last_value = jnp.asarray(last_value)
        gamma = jnp.asarray(gamma)
        gae_lambda = jnp.asarray(gae_lambda)

        next_values = jnp.concatenate([self.values[1:], last_value[None]], axis=0)
        next_non_terminals = 1.0 - self.dones.astype(float)
        deltas = self.rewards + gamma * next_values * next_non_terminals - self.values
        discounts = gamma * gae_lambda * next_non_terminals

        def scan_fn(
            carry: Float[Array, ""], x: tuple[Float[Array, ""], Float[Array, ""]]
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            delta, discount = x
            advantage = delta + discount * carry
            return advantage, advantage

        _, advantages = lax.scan(
            scan_fn, jnp.array(0.0), (deltas, discounts), reverse=True
        )
        returns = advantages + self.values

        return dataclasses.replace(self, advantages=advantages, returns=returns)

    def batches(
        self,
        batch_size: int,
        *,
        key: Key | None = None,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> RolloutBuffer[StateType, ActType, ObsType]:
        flat_self = self.flatten_axes(batch_axes)

        total = flat_self.rewards.shape[0]
        indices = jnp.arange(total) if key is None else jr.permutation(key, total)

        if total % batch_size != 0:
            total_trim = total - (total % batch_size)
            indices = indices[:total_trim]

        indices = indices.reshape(-1, batch_size)

        return jax.tree.map(lambda x: jnp.take(x, indices, axis=0), flat_self)

    def sample(
        self,
        batch_size: int,
        *,
        key: Key,
        batch_axes: tuple[int, ...] | int | None = None,
    ) -> RolloutBuffer[StateType, ActType, ObsType]:
        flat_self = self.flatten_axes(batch_axes)

        total = flat_self.rewards.shape[0]
        if batch_size > total:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from total={total} elements."
            )

        indices = jr.choice(key, total, shape=(batch_size,), replace=False)

        return jax.tree.map(lambda x: jnp.take(x, indices, axis=0), flat_self)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rewards.shape
