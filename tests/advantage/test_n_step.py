import jax.numpy as jnp
import pytest

from lerax.advantage import GAE, BootstrappedReturn, NStepReturn


def test_n_step_one_matches_gae_lambda_zero():
    rewards = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([0.5, 0.25, 1.0])
    dones = jnp.array([False, False, True])
    last_value = jnp.array(4.0)

    expected = GAE(gamma=0.9, lam=0.0)(rewards, values, dones, last_value)
    actual = NStepReturn(n=1, gamma=0.9)(rewards, values, dones, last_value)

    assert jnp.allclose(actual[0], expected[0])
    assert jnp.allclose(actual[1], expected[1])


def test_large_n_step_matches_bootstrapped_return():
    rewards = jnp.array([1.0, -0.5, 2.0, 0.25])
    values = jnp.array([0.3, 0.1, -0.2, 0.4])
    dones = jnp.array([False, True, False, False])
    last_value = jnp.array(0.75)

    expected = BootstrappedReturn(gamma=0.9)(rewards, values, dones, last_value)
    actual = NStepReturn(n=10, gamma=0.9)(rewards, values, dones, last_value)

    assert jnp.allclose(actual[0], expected[0])
    assert jnp.allclose(actual[1], expected[1])


def test_n_step_stops_at_episode_boundary():
    rewards = jnp.ones(4)
    values = jnp.zeros(4)
    dones = jnp.array([False, True, False, False])

    _, returns = NStepReturn(n=3, gamma=1.0)(rewards, values, dones, jnp.array(5.0))

    assert jnp.allclose(returns, jnp.array([2.0, 1.0, 7.0, 6.0]))


def test_n_step_requires_positive_horizon():
    with pytest.raises(ValueError, match="n must be at least 1"):
        NStepReturn(n=0)
