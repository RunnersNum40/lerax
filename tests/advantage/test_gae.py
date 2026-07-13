import equinox as eqx
import jax.numpy as jnp

from lerax.advantage import GAE
from lerax.buffer import RolloutBuffer


def reference_gae(rewards, values, dones, last_value, gamma, lam):
    advantages = jnp.zeros_like(rewards)
    next_advantage = jnp.zeros((), dtype=rewards.dtype)

    for index in range(rewards.shape[0] - 1, -1, -1):
        next_value = last_value if index == rewards.shape[0] - 1 else values[index + 1]
        not_done = 1.0 - dones[index].astype(float)
        delta = rewards[index] + gamma * next_value * not_done - values[index]
        next_advantage = delta + gamma * lam * not_done * next_advantage
        advantages = advantages.at[index].set(next_advantage)

    return advantages, advantages + values


def test_gae_matches_reference_and_rollout_buffer():
    rewards = jnp.array([1.0, -0.5, 2.0, 0.25])
    values = jnp.array([0.3, 0.1, -0.2, 0.4])
    dones = jnp.array([False, True, False, False])
    last_value = jnp.array(0.75)
    gamma = 0.9
    lam = 0.8

    expected_advantages, expected_returns = reference_gae(
        rewards, values, dones, last_value, gamma, lam
    )
    advantages, returns = GAE(gamma=gamma, lam=lam)(rewards, values, dones, last_value)
    buffer = RolloutBuffer(
        observations=jnp.zeros((4, 1)),
        actions=jnp.zeros(4, dtype=int),
        rewards=rewards,
        dones=dones,
        log_probs=jnp.zeros(4),
        values=values,
        states=None,
    ).compute_returns_and_advantages(last_value, lam, gamma)

    assert jnp.allclose(advantages, expected_advantages)
    assert jnp.allclose(returns, expected_returns)
    assert jnp.allclose(buffer.advantages, expected_advantages)
    assert jnp.allclose(buffer.returns, expected_returns)


def test_gae_lambda_zero_is_one_step_td():
    rewards = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([0.5, 0.25, 1.0])
    dones = jnp.array([False, False, True])
    last_value = jnp.array(4.0)
    gamma = 0.9

    advantages, _ = GAE(gamma=gamma, lam=0.0)(rewards, values, dones, last_value)
    next_values = jnp.array([0.25, 1.0, 4.0])
    expected = rewards + gamma * next_values * (~dones) - values

    assert jnp.allclose(advantages, expected)


def test_gae_done_cuts_advantage_propagation():
    rewards = jnp.ones(4)
    values = jnp.zeros(4)
    dones = jnp.array([False, True, False, False])

    advantages, returns = GAE(gamma=1.0, lam=1.0)(
        rewards, values, dones, jnp.array(0.0)
    )

    expected = jnp.array([2.0, 1.0, 2.0, 1.0])
    assert jnp.allclose(advantages, expected)
    assert jnp.allclose(returns, expected)


def test_gae_lambda_one_bootstraps_incomplete_segment():
    rewards = jnp.zeros(3)
    values = jnp.zeros(3)
    dones = jnp.zeros(3, dtype=bool)

    _, returns = GAE(gamma=1.0, lam=1.0)(rewards, values, dones, jnp.array(3.0))

    assert jnp.allclose(returns, jnp.full(3, 3.0))


def test_gae_is_jittable():
    estimator = eqx.filter_jit(GAE(gamma=jnp.array(0.99), lam=jnp.array(0.95)))
    advantages, returns = estimator(
        jnp.ones(8),
        jnp.zeros(8),
        jnp.zeros(8, dtype=bool),
        jnp.array(0.0),
    )

    assert advantages.shape == (8,)
    assert returns.shape == (8,)
    assert jnp.all(jnp.isfinite(advantages))
    assert jnp.all(jnp.isfinite(returns))
