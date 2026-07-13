import equinox as eqx
import jax.numpy as jnp

from lerax.advantage import GAE, BootstrappedReturn, discounted_returns


def test_discounted_returns_bootstraps_and_stops_at_episode_boundaries():
    returns = eqx.filter_jit(discounted_returns)(
        jnp.array([1.0, 2.0, 3.0, 4.0]),
        jnp.array([False, True, False, False]),
        jnp.array(5.0),
        jnp.array(1.0),
    )

    assert jnp.allclose(returns, jnp.array([3.0, 2.0, 12.0, 9.0]))


def test_bootstrapped_return_matches_gae_lambda_one():
    rewards = jnp.array([1.0, -0.5, 2.0, 0.25])
    values = jnp.array([0.3, 0.1, -0.2, 0.4])
    dones = jnp.array([False, True, False, False])
    last_value = jnp.array(0.75)

    expected = GAE(gamma=0.9, lam=1.0)(rewards, values, dones, last_value)
    actual = BootstrappedReturn(gamma=0.9)(rewards, values, dones, last_value)

    assert jnp.allclose(actual[0], expected[0])
    assert jnp.allclose(actual[1], expected[1])
