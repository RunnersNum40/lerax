from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Key, Scalar

from lerax.buffer import ReplayBuffer
from lerax.policy import AbstractStatefulDQNPolicy
from lerax.utils import filter_scan

from .off_policy import AbstractOffPolicyAlgorithm, IterationCarry, StepCarry


class DQNStats(eqx.Module):
    q_loss: Float[Array, ""]
    td_error: Float[Array, ""]
    mean_q: Float[Array, ""]


class DQN[PolicyType: AbstractStatefulDQNPolicy](
    AbstractOffPolicyAlgorithm[PolicyType]
):
    optimizer: optax.GradientTransformation

    gamma: float
    tau: float
    learning_starts: int
    target_update_period: int
    max_grad_norm: float

    num_envs: int
    num_steps: int
    gradient_steps: int
    batch_size: int
    buffer_size: int

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        tau: float = 1.0,
        learning_starts: int = 128,
        target_update_period: int = 1000,
        max_grad_norm: float = 10.0,
        num_envs: int = 1,
        num_steps: int = 1,
        gradient_steps: int = 1,
        batch_size: int = 32,
        buffer_size: int = 2**20,
        learning_rate: float = 1e-4,
    ):
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.target_update_period = target_update_period
        self.max_grad_norm = max_grad_norm

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        adam = optax.inject_hyperparams(optax.adam)(learning_rate)
        clip = optax.clip_by_global_norm(self.max_grad_norm)
        self.optimizer = optax.chain(clip, adam)

    def per_step(self, step_carry: StepCarry[PolicyType]) -> StepCarry[PolicyType]:
        return step_carry

    def per_iteration(
        self, iteration_carry: IterationCarry[PolicyType]
    ) -> IterationCarry[PolicyType]:
        if self.target_update_period <= 0:
            return iteration_carry

        updates = iteration_carry.iteration_count * self.gradient_steps

        policy = iteration_carry.policy
        current_dynamic = eqx.filter(policy.q_network, eqx.is_inexact_array)
        target_dynamic, target_static = eqx.partition(
            policy.target_q_network, eqx.is_inexact_array
        )

        updated_dynamic = jax.tree.map(
            lambda q, tq: self.tau * q + (1 - self.tau) * tq,
            current_dynamic,
            target_dynamic,
        )

        should_update = jnp.equal(jnp.remainder(updates, self.target_update_period), 0)
        next_target_dynamic = lax.cond(
            should_update, lambda: updated_dynamic, lambda: target_dynamic
        )

        target_q_network = eqx.combine(next_target_dynamic, target_static)
        policy = eqx.tree_at(lambda p: p.target_q_network, policy, target_q_network)

        return IterationCarry(
            iteration_carry.iteration_count,
            iteration_carry.step_carry,
            policy,
            iteration_carry.opt_state,
        )

    # Needs to be static so the first argument can be a policy
    # eqx.filter_value_and_grad doesn't support argnums
    @staticmethod
    def dqn_loss(
        policy: PolicyType,
        sample: ReplayBuffer,
        gamma: float,
    ) -> tuple[Float[Array, ""], DQNStats]:
        q_network = policy.q_network
        target_dynamic, target_static = eqx.partition(
            policy.target_q_network, eqx.is_inexact_array
        )
        target_dynamic = lax.stop_gradient(target_dynamic)
        target_q_network = eqx.combine(target_dynamic, target_static)

        target_max = jax.vmap(target_q_network.q_values)(
            sample.next_states, sample.next_observations
        )[1].max(axis=-1)
        td_target = sample.rewards + gamma * target_max * (
            1.0 - sample.dones.astype(float)
        )

        current_q = jnp.take_along_axis(
            jax.vmap(q_network.q_values)(sample.states, sample.observations)[1],
            sample.actions.astype(int)[..., None],
            axis=-1,
        ).squeeze(-1)
        q_loss = optax.huber_loss(current_q, td_target).mean()

        stats = DQNStats(
            q_loss=q_loss,
            td_error=jnp.mean(jnp.abs(td_target - current_q)),
            mean_q=jnp.mean(current_q),
        )

        return q_loss, stats

    dqn_loss_grad = staticmethod(eqx.filter_value_and_grad(dqn_loss, has_aux=True))

    def train_sample(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        sample: ReplayBuffer,
    ) -> tuple[PolicyType, optax.OptState, DQNStats]:
        (_, stats), grads = self.dqn_loss_grad(policy, sample, self.gamma)

        updates, new_opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, updates)

        return policy, new_opt_state, stats

    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        *,
        key: Key,
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        def sample_scan(
            carry: tuple[PolicyType, optax.OptState], key: Key
        ) -> tuple[tuple[PolicyType, optax.OptState], DQNStats]:
            policy, opt_state = carry
            policy, opt_state, stats = self.train_sample(
                policy, opt_state, buffer.sample(self.batch_size, key=key)
            )
            return (policy, opt_state), stats

        (policy, opt_state), stats = filter_scan(
            sample_scan, (policy, opt_state), jr.split(key, self.gradient_steps)
        )

        stats = jax.tree.map(jnp.mean, stats)
        log = {
            "q_loss": stats.q_loss,
            "td_error": stats.td_error,
            "mean_q": stats.mean_q,
        }
        return policy, opt_state, log
