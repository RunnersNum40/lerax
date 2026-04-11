from jax import numpy as jnp
from jax import random as jr

from lerax.algorithm import PPO
from lerax.callback import CallbackList
from lerax.curriculum import ScheduledCurriculum, linear_schedule, step_schedule
from lerax.env.classic_control import Pendulum
from lerax.policy import MLPActorCriticPolicy


def test_scheduled_curriculum_linear():
    """Test that linear curriculum modifies env parameters during training."""
    policy_key, learn_key = jr.split(jr.key(0), 2)

    env = Pendulum()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)

    curriculum = ScheduledCurriculum(
        where=lambda env: env.m,
        schedule_fn=linear_schedule(start=0.5, end=2.0, total=10),
    )

    algo = PPO(num_envs=1, num_steps=32)
    trained_policy = algo.learn(
        env, policy, total_timesteps=320, key=learn_key, callback=curriculum
    )

    assert trained_policy is not None


def test_scheduled_curriculum_step():
    """Test step schedule with discrete difficulty levels."""
    policy_key, learn_key = jr.split(jr.key(1), 2)

    env = Pendulum()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)

    curriculum = ScheduledCurriculum(
        where=lambda env: env.g,
        schedule_fn=step_schedule(
            values=[5.0, 7.0, 9.8],
            boundaries=[3, 7],
        ),
    )

    algo = PPO(num_envs=1, num_steps=32)
    trained_policy = algo.learn(
        env, policy, total_timesteps=320, key=learn_key, callback=curriculum
    )

    assert trained_policy is not None


def test_scheduled_curriculum_with_other_callbacks():
    """Test that curriculum composes with other callbacks via CallbackList."""
    policy_key, learn_key = jr.split(jr.key(2), 2)

    env = Pendulum()
    policy = MLPActorCriticPolicy(env=env, key=policy_key)

    curriculum = ScheduledCurriculum(
        where=lambda env: env.m,
        schedule_fn=linear_schedule(start=0.5, end=2.0, total=10),
    )

    callbacks = CallbackList(callbacks=[curriculum])

    algo = PPO(num_envs=1, num_steps=32)
    trained_policy = algo.learn(
        env, policy, total_timesteps=320, key=learn_key, callback=callbacks
    )

    assert trained_policy is not None


def test_schedule_functions():
    """Test schedule helper functions produce correct values."""
    # Linear
    lin = linear_schedule(start=0.0, end=1.0, total=100)
    assert jnp.isclose(lin(jnp.array(0)), 0.0)
    assert jnp.isclose(lin(jnp.array(50)), 0.5)
    assert jnp.isclose(lin(jnp.array(100)), 1.0)
    assert jnp.isclose(lin(jnp.array(200)), 1.0)  # clamped

    # Step
    stp = step_schedule(values=[1.0, 2.0, 3.0], boundaries=[10, 20])
    assert jnp.isclose(stp(jnp.array(0)), 1.0)
    assert jnp.isclose(stp(jnp.array(9)), 1.0)
    assert jnp.isclose(stp(jnp.array(10)), 2.0)
    assert jnp.isclose(stp(jnp.array(19)), 2.0)
    assert jnp.isclose(stp(jnp.array(20)), 3.0)
