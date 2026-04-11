from __future__ import annotations

import equinox as eqx
import jax
import optax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int, Key, Scalar

from lerax.buffer import ReplayBuffer
from lerax.callback import (
    AbstractCallback,
    AbstractCallbackState,
    AbstractCallbackStepState,
    IterationContext,
    ResetContext,
    StepContext,
)
from lerax.env import AbstractEnvLike, AbstractEnvLikeState
from lerax.policy import AbstractPolicyState
from lerax.policy.sac.discrete import MLPDiscreteSACPolicy
from lerax.utils import filter_scan, polyak_average

from .base_algorithm import AbstractAlgorithm, AbstractAlgorithmState, AbstractStepState


class DiscreteQNetwork(eqx.Module):
    """
    Discrete Q-network for SAC-Discrete.

    Maps observations to Q-values for each discrete action.

    Attributes:
        mlp: The MLP that processes observations.

    Args:
        observation_size: Dimensionality of flat observations.
        num_actions: Number of discrete actions.
        width_size: Width of the hidden layers.
        depth: Number of hidden layers.
        key: JAX PRNG key for parameter initialization.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        *,
        width_size: int = 256,
        depth: int = 2,
        key: Key[Array, ""],
    ):
        self.mlp = eqx.nn.MLP(
            in_size=observation_size,
            out_size=num_actions,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(
        self,
        observation: Float[Array, " obs_dim"],
    ) -> Float[Array, " num_actions"]:
        """Compute Q-values for all actions given an observation."""
        return self.mlp(observation.ravel())


class SACDiscreteStepState[PolicyType: MLPDiscreteSACPolicy](AbstractStepState):
    """
    Step-level state for SAC-Discrete.

    Attributes:
        env_state: The state of the environment.
        policy_state: The state of the policy.
        callback_state: The state of the callback for this step.
        buffer: The replay buffer storing experience.
    """

    env_state: AbstractEnvLikeState
    policy_state: AbstractPolicyState
    callback_state: AbstractCallbackStepState
    buffer: ReplayBuffer

    @classmethod
    def initial(
        cls,
        size: int,
        env: AbstractEnvLike,
        policy: PolicyType,
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> SACDiscreteStepState[PolicyType]:
        """Initialize the step state with an empty replay buffer."""
        env_key, policy_key = jr.split(key, 2)
        env_state = env.initial(key=env_key)
        policy_state = policy.reset(key=policy_key)
        callback_state = callback.step_reset(ResetContext(locals()), key=key)
        buffer = ReplayBuffer(
            size,
            env.observation_space,
            env.action_space,
            policy_state,  # ty: ignore[invalid-argument-type]
        )
        return cls(env_state, policy_state, callback_state, buffer)  # ty: ignore[invalid-argument-type]


class SACDiscreteState[PolicyType: MLPDiscreteSACPolicy](
    AbstractAlgorithmState[PolicyType]
):
    """
    Iteration-level state for SAC-Discrete.

    Attributes:
        iteration_count: The current iteration count.
        step_state: The step-level state.
        env: The environment being used.
        policy: The policy being trained.
        opt_state: The actor optimizer state.
        callback_state: The callback state.
        qf1: First discrete Q-network.
        qf2: Second discrete Q-network.
        qf1_target: Target for first Q-network.
        qf2_target: Target for second Q-network.
        q_opt_state: Optimizer state for Q-networks.
        log_alpha: Log of the entropy coefficient.
        alpha_opt_state: Optimizer state for entropy coefficient.
        target_entropy: Target entropy value.
    """

    iteration_count: Int[Array, ""]
    step_state: SACDiscreteStepState[PolicyType]
    env: AbstractEnvLike
    policy: PolicyType
    opt_state: optax.OptState
    callback_state: AbstractCallbackState
    qf1: DiscreteQNetwork
    qf2: DiscreteQNetwork
    qf1_target: DiscreteQNetwork
    qf2_target: DiscreteQNetwork
    q_opt_state: optax.OptState
    log_alpha: Float[Array, ""]
    alpha_opt_state: optax.OptState
    target_entropy: Float[Array, ""]


class SACDiscrete[PolicyType: MLPDiscreteSACPolicy](
    AbstractAlgorithm[PolicyType, SACDiscreteState[PolicyType]]
):
    """
    Soft Actor-Critic for discrete action spaces (SAC-Discrete).

    Uses twin Q-networks that output Q-values for all actions, a
    categorical policy, and entropy regularization over the discrete
    action distribution. Based on the SAC-Discrete paper by
    Christodoulou (2019).

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
        autotune: Whether to automatically tune the entropy coefficient.
        initial_alpha: Initial entropy coefficient value.
        target_entropy_scale: Scale for target entropy relative to
            ``log(num_actions)``.
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
        autotune: Whether to automatically tune entropy coefficient.
        initial_alpha: Initial entropy coefficient value.
        target_entropy_scale: Scale for target entropy.
        policy_lr: Learning rate for the actor.
        q_lr: Learning rate for Q-networks.
        alpha_lr: Learning rate for the entropy coefficient.
            Defaults to ``q_lr`` when ``None``.
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
    autotune: bool
    initial_alpha: float
    target_entropy_scale: float

    q_width_size: int
    q_depth: int

    def __init__(
        self,
        *,
        buffer_size: int = 1_000_000,
        gamma: float = 0.99,
        learning_starts: int = 20_000,
        num_envs: int = 1,
        num_steps: int = 4,
        batch_size: int = 64,
        tau: float = 1.0,
        autotune: bool = True,
        initial_alpha: float = 0.2,
        target_entropy_scale: float = 0.89,
        policy_lr: optax.ScalarOrSchedule = 3e-4,
        q_lr: optax.ScalarOrSchedule = 3e-4,
        alpha_lr: optax.ScalarOrSchedule | None = None,
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
        self.autotune = autotune
        self.initial_alpha = initial_alpha
        self.target_entropy_scale = target_entropy_scale

        self.q_width_size = q_width_size
        self.q_depth = q_depth

        self.optimizer = optax.inject_hyperparams(optax.adam)(policy_lr)
        self.q_optimizer = optax.inject_hyperparams(optax.adam)(q_lr)
        self.alpha_optimizer = optax.inject_hyperparams(optax.adam)(
            alpha_lr if alpha_lr is not None else q_lr
        )

    def num_iterations(self, total_timesteps: int) -> int:
        return total_timesteps // (self.num_envs * self.num_steps)

    def step(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        state: SACDiscreteStepState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> SACDiscreteStepState[PolicyType]:
        """Perform a single environment step and store in replay buffer."""
        (
            action_key,
            transition_key,
            observation_key,
            reward_key,
            terminal_key,
            next_observation_key,
            env_reset_key,
            policy_reset_key,
            callback_key,
        ) = jr.split(key, 9)

        observation = env.observation(state.env_state, key=observation_key)
        policy_state, action = policy(state.policy_state, observation, key=action_key)  # ty: ignore[invalid-argument-type]

        next_env_state = env.transition(state.env_state, action, key=transition_key)

        reward = env.reward(state.env_state, action, next_env_state, key=reward_key)
        termination = env.terminal(next_env_state, key=terminal_key)
        truncation = env.truncate(next_env_state)
        done = termination | truncation
        timeout = truncation & ~termination
        next_observation = env.observation(next_env_state, key=next_observation_key)

        next_env_state = lax.cond(
            done, lambda: env.initial(key=env_reset_key), lambda: next_env_state
        )
        next_policy_state = lax.cond(
            done, lambda: policy.reset(key=policy_reset_key), lambda: policy_state
        )

        replay_buffer = state.buffer.add(
            observation,
            next_observation,
            action,
            reward,
            done,
            timeout,
            state.policy_state,
            policy_state,
        )

        callback_state = callback.on_step(
            StepContext(state.callback_state, env, policy, done, reward, locals()),
            key=callback_key,
        )

        return SACDiscreteStepState(
            next_env_state, next_policy_state, callback_state, replay_buffer
        )

    def collect_learning_starts(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: SACDiscreteStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> SACDiscreteStepState[PolicyType]:
        """Collect random initial experience before training begins."""

        def scan_step(
            carry: SACDiscreteStepState, key: Key[Array, ""]
        ) -> tuple[SACDiscreteStepState, None]:
            carry = self.step(env, policy, carry, key=key, callback=callback)
            return carry, None

        step_state, _ = filter_scan(
            scan_step, step_state, jr.split(key, self.learning_starts)
        )
        return step_state

    def collect_rollout(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        step_state: SACDiscreteStepState[PolicyType],
        callback: AbstractCallback,
        key: Key[Array, ""],
    ) -> SACDiscreteStepState[PolicyType]:
        """Collect a rollout of experience into the replay buffer."""

        def scan_step(
            carry: SACDiscreteStepState, key: Key[Array, ""]
        ) -> tuple[SACDiscreteStepState, None]:
            carry = self.step(env, policy, carry, key=key, callback=callback)
            return carry, None

        step_state, _ = filter_scan(
            scan_step, step_state, jr.split(key, self.num_steps)
        )
        return step_state

    def reset(
        self,
        env: AbstractEnvLike,
        policy: PolicyType,
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> SACDiscreteState[PolicyType]:
        init_key, starts_key, callback_key, qf1_key, qf2_key = jr.split(key, 5)

        if self.num_envs == 1:
            step_state = SACDiscreteStepState.initial(
                self.buffer_size, env, policy, callback, init_key
            )
            step_state = self.collect_learning_starts(
                env, policy, step_state, callback, starts_key
            )
        else:
            step_state = jax.vmap(
                SACDiscreteStepState.initial, in_axes=(None, None, None, None, 0)
            )(
                self.buffer_size // self.num_envs,
                env,
                policy,
                callback,
                jr.split(init_key, self.num_envs),
            )
            step_state = jax.vmap(
                self.collect_learning_starts, in_axes=(None, None, 0, None, 0)
            )(env, policy, step_state, callback, jr.split(starts_key, self.num_envs))

        callback_state = callback.reset(ResetContext(locals()), key=callback_key)

        num_actions = env.action_space.flat_size
        observation_size = env.observation_space.flat_size

        qf1 = DiscreteQNetwork(
            observation_size,
            num_actions,
            width_size=self.q_width_size,
            depth=self.q_depth,
            key=qf1_key,
        )
        qf2 = DiscreteQNetwork(
            observation_size,
            num_actions,
            width_size=self.q_width_size,
            depth=self.q_depth,
            key=qf2_key,
        )

        q_params = (
            eqx.filter(qf1, eqx.is_inexact_array),
            eqx.filter(qf2, eqx.is_inexact_array),
        )
        q_opt_state = self.q_optimizer.init(q_params)

        log_alpha = jnp.log(jnp.array(self.initial_alpha))
        alpha_opt_state = self.alpha_optimizer.init(log_alpha)
        target_entropy = jnp.array(
            -self.target_entropy_scale * jnp.log(1.0 / num_actions)
        )

        return SACDiscreteState(
            jnp.array(0, dtype=int),
            step_state,
            env,
            policy,
            self.optimizer.init(eqx.filter(policy, eqx.is_inexact_array)),
            callback_state,
            qf1=qf1,
            qf2=qf2,
            qf1_target=qf1,
            qf2_target=qf2,
            q_opt_state=q_opt_state,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
            target_entropy=target_entropy,
        )

    def iteration(
        self,
        state: SACDiscreteState[PolicyType],
        *,
        key: Key[Array, ""],
        callback: AbstractCallback,
    ) -> SACDiscreteState[PolicyType]:
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
        ) = self.sac_discrete_train(
            state.policy,
            state.opt_state,
            step_state.buffer,
            state.qf1,
            state.qf2,
            state.qf1_target,
            state.qf2_target,
            state.q_opt_state,
            state.log_alpha,
            state.alpha_opt_state,
            state.target_entropy,
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
                    self,
                    locals(),
                ),
                key=callback_key,
            )
        )

        state, new_cb = callback.apply_curriculum(state, state.callback_state)
        state = state.with_callback_states(new_cb)
        return self.per_iteration(state)

    def per_iteration(
        self, state: SACDiscreteState[PolicyType]
    ) -> SACDiscreteState[PolicyType]:
        """Polyak-average target Q-networks."""

        return eqx.tree_at(
            lambda s: (s.qf1_target, s.qf2_target),
            state,
            (
                polyak_average(state.qf1, state.qf1_target, self.tau),
                polyak_average(state.qf2, state.qf2_target, self.tau),
            ),
        )

    @staticmethod
    def q_loss(
        q_params: tuple[DiscreteQNetwork, DiscreteQNetwork],
        batch: ReplayBuffer,
        target: Float[Array, " batch_size"],
    ) -> Float[Array, ""]:
        """Compute combined MSE loss for twin discrete Q-networks."""
        qf1, qf2 = q_params
        qf1_all = jax.vmap(qf1)(batch.observations)
        qf2_all = jax.vmap(qf2)(batch.observations)
        actions = batch.actions.astype(int)
        qf1_values = qf1_all[jnp.arange(actions.shape[0]), actions]
        qf2_values = qf2_all[jnp.arange(actions.shape[0]), actions]
        qf1_loss = jnp.mean(jnp.square(qf1_values - target)) / 2
        qf2_loss = jnp.mean(jnp.square(qf2_values - target)) / 2
        return qf1_loss + qf2_loss

    q_loss_grad = staticmethod(eqx.filter_value_and_grad(q_loss))

    @staticmethod
    def actor_loss(
        policy: PolicyType,
        batch: ReplayBuffer,
        qf1: DiscreteQNetwork,
        qf2: DiscreteQNetwork,
        alpha: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Compute actor loss over discrete action distribution."""

        def single_loss(observation):
            _, probs, log_probs = policy.action_probs_and_log_probs(None, observation)
            q1 = qf1(observation)
            q2 = qf2(observation)
            min_q = jnp.minimum(q1, q2)
            return jnp.sum(probs * (alpha * log_probs - min_q))

        return jnp.mean(jax.vmap(single_loss)(batch.observations))

    actor_loss_grad = staticmethod(eqx.filter_value_and_grad(actor_loss))

    @staticmethod
    def alpha_loss(
        log_alpha: Float[Array, ""],
        action_probs: Float[Array, "batch_size num_actions"],
        log_probs: Float[Array, "batch_size num_actions"],
        target_entropy: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Compute entropy coefficient loss for discrete actions."""
        alpha = jnp.exp(log_alpha)
        return jnp.mean(
            jnp.sum(
                action_probs * (-alpha * (log_probs + target_entropy)),
                axis=-1,
            )
        )

    alpha_loss_grad = staticmethod(eqx.filter_value_and_grad(alpha_loss))

    def sac_discrete_train(
        self,
        policy: PolicyType,
        opt_state: optax.OptState,
        buffer: ReplayBuffer,
        qf1: DiscreteQNetwork,
        qf2: DiscreteQNetwork,
        qf1_target: DiscreteQNetwork,
        qf2_target: DiscreteQNetwork,
        q_opt_state: optax.OptState,
        log_alpha: Float[Array, ""],
        alpha_opt_state: optax.OptState,
        target_entropy: Float[Array, ""],
        *,
        key: Key[Array, ""],
    ) -> tuple[
        PolicyType,
        optax.OptState,
        DiscreteQNetwork,
        DiscreteQNetwork,
        optax.OptState,
        Float[Array, ""],
        optax.OptState,
        dict[str, Scalar],
    ]:
        sample_key, train_key = jr.split(key, 2)
        batch = buffer.sample(self.batch_size, key=sample_key)
        alpha = jnp.exp(log_alpha)

        # Compute Q-targets using expectation over next policy
        def compute_target(next_obs, reward, done, timeout):
            _, next_probs, next_log_probs = policy.action_probs_and_log_probs(
                None, next_obs
            )
            q1_next = qf1_target(next_obs)
            q2_next = qf2_target(next_obs)
            min_q_next = jnp.minimum(q1_next, q2_next)
            v_next = jnp.sum(next_probs * (min_q_next - alpha * next_log_probs))
            not_terminal = (~done | timeout).astype(float)
            return reward + self.gamma * v_next * not_terminal

        targets = jax.vmap(compute_target)(
            batch.next_observations, batch.rewards, batch.dones, batch.timeouts
        )

        # Update twin Q-networks
        q_loss_val, q_grads = self.q_loss_grad((qf1, qf2), batch, targets)
        q_updates, q_opt_state = self.q_optimizer.update(
            q_grads,
            q_opt_state,
            (
                eqx.filter(qf1, eqx.is_inexact_array),
                eqx.filter(qf2, eqx.is_inexact_array),
            ),
        )
        qf1, qf2 = eqx.apply_updates((qf1, qf2), q_updates)

        # Update actor
        _, a_grads = self.actor_loss_grad(policy, batch, qf1, qf2, alpha)
        a_updates, opt_state = self.optimizer.update(
            a_grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, a_updates)

        # Update alpha
        if self.autotune:
            _, all_probs, all_log_probs = jax.vmap(
                policy.action_probs_and_log_probs, in_axes=(None, 0)
            )(None, batch.observations)

            al_loss, al_grads = self.alpha_loss_grad(
                log_alpha, all_probs, all_log_probs, target_entropy
            )
            al_updates, alpha_opt_state = self.alpha_optimizer.update(
                al_grads, alpha_opt_state
            )
            log_alpha = optax.apply_updates(log_alpha, al_updates)  # ty: ignore[invalid-assignment]

        log: dict[str, Scalar] = {"q_loss": q_loss_val}
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
