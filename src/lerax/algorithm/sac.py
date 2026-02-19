from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int, Key, Scalar

from lerax.buffer import ReplayBuffer
from lerax.callback import AbstractCallback, IterationContext
from lerax.env import AbstractEnvLike
from lerax.policy.sac import AbstractSACPolicy
from lerax.utils import filter_cond

from .off_policy import (
    AbstractOffPolicyAlgorithm,
    AbstractOffPolicyState,
    AbstractOffPolicyStepState,
)


class SoftQNetwork(eqx.Module):
    """
    Soft Q-network for SAC.

    Maps concatenated (observation, action) pairs to scalar Q-values
    using an MLP.

    Attributes:
        mlp: The MLP that processes the concatenated input.

    Args:
        observation_size: Dimensionality of flat observations.
        action_size: Dimensionality of flat actions.
        width_size: Width of the hidden layers.
        depth: Number of hidden layers.
        key: JAX PRNG key for parameter initialization.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        width_size: int = 256,
        depth: int = 2,
        key: Key[Array, ""],
    ):
        self.mlp = eqx.nn.MLP(
            in_size=observation_size + action_size,
            out_size="scalar",
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(
        self,
        observation: Float[Array, " obs_dim"],
        action: Float[Array, " act_dim"],
    ) -> Float[Array, ""]:
        """Compute Q-value for an observation-action pair."""
        inputs = jnp.concatenate(
            [observation.ravel(), action.ravel()],
        )
        return self.mlp(inputs)


class SACState[PolicyType: AbstractSACPolicy](AbstractOffPolicyState[PolicyType]):
    """
    State for SAC algorithms.

    Extends the off-policy state with twin Q-networks, target Q-networks,
    and entropy coefficient state.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The current step state.
        env: The environment being used.
        policy: The policy being trained.
        opt_state: The actor optimizer state.
        callback_state: The callback state.
        qf1: First Q-network.
        qf2: Second Q-network.
        qf1_target: Target for first Q-network.
        qf2_target: Target for second Q-network.
        q_opt_state: Optimizer state for Q-networks.
        log_alpha: Log of the entropy coefficient.
        alpha_opt_state: Optimizer state for entropy coefficient.
        target_entropy: Target entropy value.
    """

    qf1: SoftQNetwork
    qf2: SoftQNetwork
    qf1_target: SoftQNetwork
    qf2_target: SoftQNetwork
    q_opt_state: optax.OptState
    log_alpha: Float[Array, ""]
    alpha_opt_state: optax.OptState
    target_entropy: Float[Array, ""]


class SAC[PolicyType: AbstractSACPolicy](AbstractOffPolicyAlgorithm[PolicyType]):
    """
    Soft Actor-Critic (SAC) algorithm.

    A maximum-entropy off-policy algorithm for continuous action spaces.
    Uses twin Q-networks with target networks, delayed policy updates,
    and automatic entropy coefficient tuning.

    Attributes:
        optimizer: The actor optimizer.
        q_optimizer: The Q-network optimizer.
        alpha_optimizer: The entropy coefficient optimizer.
        buffer_size: The size of the replay buffer.
        gamma: Discount factor for future rewards.
        learning_starts: Number of initial steps to collect before training.
        num_envs: Number of parallel environments.
        num_steps: Number of steps per iteration.
        batch_size: Batch size for training.
        tau: Soft update coefficient for target networks.
        policy_frequency: How often to update the actor (relative to Q updates).
        autotune: Whether to automatically tune the entropy coefficient.
        initial_alpha: Initial entropy coefficient value.
        q_width_size: Width of Q-network hidden layers.
        q_depth: Depth of Q-network hidden layers.

    Args:
        buffer_size: The size of the replay buffer.
        gamma: Discount factor for future rewards.
        learning_starts: Number of initial steps before training.
        num_envs: Number of parallel environments.
        num_steps: Number of steps per iteration.
        batch_size: Batch size for training.
        tau: Soft update coefficient for target networks.
        policy_frequency: How often to update the actor.
        autotune: Whether to automatically tune entropy coefficient.
        initial_alpha: Initial entropy coefficient value.
        policy_lr: Learning rate for the actor.
        q_lr: Learning rate for Q-networks.
        q_width_size: Width of Q-network hidden layers.
        q_depth: Depth of Q-network hidden layers.
    """

    optimizer: optax.GradientTransformation
    q_optimizer: optax.GradientTransformation = eqx.field(static=True)
    alpha_optimizer: optax.GradientTransformation = eqx.field(static=True)

    buffer_size: int
    gamma: float
    learning_starts: int

    num_envs: int
    num_steps: int
    batch_size: int

    tau: float
    policy_frequency: int
    autotune: bool
    initial_alpha: float

    q_width_size: int
    q_depth: int

    def __init__(
        self,
        *,
        buffer_size: int = 1_000_000,
        gamma: float = 0.99,
        learning_starts: int = 100,
        num_envs: int = 4,
        num_steps: int = 1,
        batch_size: int = 256,
        tau: float = 0.005,
        policy_frequency: int = 2,
        autotune: bool = True,
        initial_alpha: float = 0.2,
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        q_width_size: int = 256,
        q_depth: int = 2,
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.learning_starts = learning_starts

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.tau = tau
        self.policy_frequency = policy_frequency
        self.autotune = autotune
        self.initial_alpha = initial_alpha

        self.q_width_size = q_width_size
        self.q_depth = q_depth

        self.optimizer = optax.adam(policy_lr)
        self.q_optimizer = optax.adam(q_lr)
        self.alpha_optimizer = optax.adam(q_lr)

    def per_step(
        self, step_state: AbstractOffPolicyStepState[PolicyType]
    ) -> AbstractOffPolicyStepState[PolicyType]:
        return step_state

    def per_iteration(
        self, state: AbstractOffPolicyState[PolicyType]
    ) -> AbstractOffPolicyState[PolicyType]:
        return _soft_update_targets(state, self.tau)

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> SACState[PolicyType]:
        base_key, qf1_key, qf2_key = jr.split(key, 3)
        base_state = super().reset(env, policy, key=base_key, callback=callback)

        observation_size = env.observation_space.flat_size
        action_size = max(env.action_space.flat_size, 1)

        qf1 = SoftQNetwork(
            observation_size,
            action_size,
            width_size=self.q_width_size,
            depth=self.q_depth,
            key=qf1_key,
        )
        qf2 = SoftQNetwork(
            observation_size,
            action_size,
            width_size=self.q_width_size,
            depth=self.q_depth,
            key=qf2_key,
        )

        qf1_target = qf1
        qf2_target = qf2

        q_params = (
            eqx.filter(qf1, eqx.is_inexact_array),
            eqx.filter(qf2, eqx.is_inexact_array),
        )
        q_opt_state = self.q_optimizer.init(q_params)

        log_alpha = jnp.log(jnp.array(self.initial_alpha))
        alpha_opt_state = self.alpha_optimizer.init(log_alpha)
        target_entropy = jnp.array(-float(action_size))

        return SACState(
            base_state.iteration_count,
            base_state.step_state,
            base_state.env,
            base_state.policy,
            base_state.opt_state,
            base_state.callback_state,
            qf1=qf1,
            qf2=qf2,
            qf1_target=qf1_target,
            qf2_target=qf2_target,
            q_opt_state=q_opt_state,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
            target_entropy=target_entropy,
        )

    def iteration(
        self,
        state: AbstractOffPolicyState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> AbstractOffPolicyState[PolicyType]:
        rollout_key, train_key, callback_key = jr.split(key, 3)

        if self.num_envs == 1:
            step_state = self.collect_rollout(
                state.env, state.policy, state.step_state, callback, rollout_key
            )
        else:
            step_state = eqx.filter_vmap(
                self.collect_rollout, in_axes=(None, None, eqx.if_array(0), None, 0)
            )(
                state.env,
                state.policy,
                state.step_state,
                callback,
                jr.split(rollout_key, self.num_envs),
            )

        (
            policy,
            opt_state,
            qf1,
            qf2,
            q_opt_state,
            log_alpha,
            alpha_opt_state,
            log,
        ) = self.sac_train(
            state.policy,
            state.opt_state,
            step_state.buffer,
            state.qf1,  # type: ignore[attr-defined]
            state.qf2,  # type: ignore[attr-defined]
            state.qf1_target,  # type: ignore[attr-defined]
            state.qf2_target,  # type: ignore[attr-defined]
            state.q_opt_state,  # type: ignore[attr-defined]
            state.log_alpha,  # type: ignore[attr-defined]
            state.alpha_opt_state,  # type: ignore[attr-defined]
            state.target_entropy,  # type: ignore[attr-defined]
            state.iteration_count,
            key=train_key,
        )

        state = state.next(step_state, policy, opt_state)

        state = eqx.tree_at(
            lambda s: (s.qf1, s.qf2, s.q_opt_state, s.log_alpha, s.alpha_opt_state),
            state,
            (qf1, qf2, q_opt_state, log_alpha, alpha_opt_state),
        )

        state = state.with_callback_states(
            callback.on_iteration(
                IterationContext(
                    state.callback_state,
                    state.step_state.callback_state,
                    state.env,
                    state.policy,
                    state.iteration_count,
                    state.opt_state,
                    log,
                    locals(),
                ),
                key=callback_key,
            )
        )

        return self.per_iteration(state)

    def train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        *,
        key: Key[Array, ""],
    ) -> tuple[PolicyType, optax.OptState, dict[str, Scalar]]:
        return policy, opt_state, {}

    @staticmethod
    def q_loss(
        q_params: tuple[SoftQNetwork, SoftQNetwork],
        batch: ReplayBuffer,
        target: Float[Array, " batch_size"],
    ) -> Float[Array, ""]:
        """Compute combined MSE loss for twin Q-networks."""
        qf1, qf2 = q_params

        qf1_values = jax.vmap(qf1)(batch.observations, batch.actions).squeeze()
        qf2_values = jax.vmap(qf2)(batch.observations, batch.actions).squeeze()

        qf1_loss = jnp.mean(jnp.square(qf1_values - target)) / 2
        qf2_loss = jnp.mean(jnp.square(qf2_values - target)) / 2
        return qf1_loss + qf2_loss

    q_loss_grad = staticmethod(eqx.filter_value_and_grad(q_loss))

    @staticmethod
    def actor_loss(
        policy: PolicyType,
        batch: ReplayBuffer,
        qf1: SoftQNetwork,
        qf2: SoftQNetwork,
        alpha: Float[Array, ""],
        keys: Key[Array, " batch"],
    ) -> Float[Array, ""]:
        """Compute actor loss: maximize Q-values minus entropy penalty."""

        def single_loss(observation, action_key):
            _, action, log_prob = policy.action_and_log_prob(
                None, observation, key=action_key
            )
            q1 = qf1(observation, action)
            q2 = qf2(observation, action)
            min_q = jnp.minimum(q1, q2)
            return alpha * log_prob - min_q

        losses = jax.vmap(single_loss)(batch.observations, keys)
        return jnp.mean(losses)

    actor_loss_grad = staticmethod(eqx.filter_value_and_grad(actor_loss))

    @staticmethod
    def alpha_loss(
        log_alpha: Float[Array, ""],
        log_probs: Float[Array, " batch_size"],
        target_entropy: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Compute entropy coefficient loss."""
        alpha = jnp.exp(log_alpha)
        return jnp.mean(-alpha * jax.lax.stop_gradient(log_probs + target_entropy))

    alpha_loss_grad = staticmethod(eqx.filter_value_and_grad(alpha_loss))

    def sac_train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        qf1: SoftQNetwork,
        qf2: SoftQNetwork,
        qf1_target: SoftQNetwork,
        qf2_target: SoftQNetwork,
        q_opt_state: optax.OptState,
        log_alpha: Float[Array, ""],
        alpha_opt_state: optax.OptState,
        target_entropy: Float[Array, ""],
        iteration_count: Int[Array, ""],
        *,
        key: Key[Array, ""],
    ) -> tuple[
        PolicyType,
        optax.OptState,
        SoftQNetwork,
        SoftQNetwork,
        optax.OptState,
        Float[Array, ""],
        optax.OptState,
        dict[str, Scalar],
    ]:
        sample_key, next_action_key, actor_key = jr.split(key, 3)
        batch = buffer.sample(self.batch_size, key=sample_key)
        alpha = jnp.exp(log_alpha)

        # Compute Q-network targets
        next_action_keys = jr.split(next_action_key, self.batch_size)

        def compute_target(next_obs, reward, done, timeout, action_key):
            _, next_action, next_log_prob = policy.action_and_log_prob(
                None, next_obs, key=action_key
            )
            q1_next = qf1_target(next_obs, next_action)
            q2_next = qf2_target(next_obs, next_action)
            min_q_next = jnp.minimum(q1_next, q2_next) - alpha * next_log_prob
            not_terminal = (~done | timeout).astype(float)
            return reward + self.gamma * min_q_next * not_terminal

        targets = jax.vmap(compute_target)(
            batch.next_observations,
            batch.rewards,
            batch.dones,
            batch.timeouts,
            next_action_keys,
        )

        # Q-network update
        q_loss, q_grads = self.q_loss_grad(  # type: ignore[missing-argument]
            (qf1, qf2),
            batch,
            targets,
        )

        q_updates, q_opt_state = self.q_optimizer.update(
            q_grads,
            q_opt_state,
            (
                eqx.filter(qf1, eqx.is_inexact_array),
                eqx.filter(qf2, eqx.is_inexact_array),
            ),
        )
        qf1, qf2 = eqx.apply_updates((qf1, qf2), q_updates)

        # Actor update (delayed)
        should_update_actor = iteration_count % self.policy_frequency == 0
        actor_keys = jr.split(actor_key, self.batch_size)

        def update_actor():
            a_loss, a_grads = self.actor_loss_grad(  # type: ignore[missing-argument]
                policy,
                batch,
                qf1,
                qf2,
                alpha,
                actor_keys,
            )
            a_updates, new_opt_state = self.optimizer.update(
                a_grads,
                opt_state,
                eqx.filter(policy, eqx.is_inexact_array),
            )
            new_policy = eqx.apply_updates(policy, a_updates)
            return new_policy, new_opt_state

        def skip_actor():
            return policy, opt_state

        policy, opt_state = filter_cond(should_update_actor, update_actor, skip_actor)

        # Alpha update (delayed, only if autotune)
        if self.autotune:

            def compute_log_probs(obs, action_key):
                _, _, lp = policy.action_and_log_prob(None, obs, key=action_key)
                return lp

            log_probs = jax.vmap(compute_log_probs)(batch.observations, actor_keys)

            def update_alpha():
                al_loss, al_grads = self.alpha_loss_grad(  # type: ignore[missing-argument]
                    log_alpha, log_probs, target_entropy
                )
                al_updates, new_alpha_opt_state = self.alpha_optimizer.update(
                    al_grads, alpha_opt_state
                )
                new_log_alpha = optax.apply_updates(log_alpha, al_updates)
                return new_log_alpha, new_alpha_opt_state

            def skip_alpha():
                return log_alpha, alpha_opt_state

            log_alpha, alpha_opt_state = filter_cond(
                should_update_actor, update_alpha, skip_alpha
            )

        log: dict[str, Scalar] = {"q_loss": q_loss}
        return (
            policy,
            opt_state,
            qf1,
            qf2,
            q_opt_state,
            log_alpha,
            alpha_opt_state,
            log,
        )


def _soft_update_targets[PolicyType: AbstractSACPolicy](
    state: AbstractOffPolicyState[PolicyType], tau: float
) -> AbstractOffPolicyState[PolicyType]:
    """Apply Polyak averaging to update target Q-networks."""

    def polyak(online, target):
        online_arrays, online_static = eqx.partition(online, eqx.is_inexact_array)
        target_arrays, _ = eqx.partition(target, eqx.is_inexact_array)

        updated_arrays = jax.tree.map(
            lambda o, t: tau * o + (1 - tau) * t,
            online_arrays,
            target_arrays,
        )
        return eqx.combine(updated_arrays, online_static)

    new_qf1_target = polyak(state.qf1, state.qf1_target)  # type: ignore[attr-defined]
    new_qf2_target = polyak(state.qf2, state.qf2_target)  # type: ignore[attr-defined]

    return eqx.tree_at(
        lambda s: (s.qf1_target, s.qf2_target),
        state,
        (new_qf1_target, new_qf2_target),
    )
