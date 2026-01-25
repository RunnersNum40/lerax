import pytest
from jax import numpy as jnp
from jax import random as jr

from lerax.distribution import Normal


class TestNormal:
    """Tests for the Normal distribution."""

    def test_init(self):
        """Test basic initialization."""
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array([1.0, 2.0])
        dist = Normal(loc=loc, scale=scale)

        assert jnp.allclose(dist.loc, loc)
        assert jnp.allclose(dist.scale, scale)

    def test_init_validates_shape_match(self):
        """Test that loc and scale must have the same shape."""
        with pytest.raises(ValueError):
            Normal(loc=jnp.array([0.0, 1.0]), scale=jnp.array([1.0]))

    def test_sample_shape(self):
        """Test that samples have the correct shape."""
        loc = jnp.array([0.0, 1.0, 2.0])
        scale = jnp.array([1.0, 1.0, 1.0])
        dist = Normal(loc=loc, scale=scale)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (3,)
        assert jnp.issubdtype(sample.dtype, jnp.floating)

    def test_sample_and_log_prob(self):
        """Test sample_and_log_prob returns consistent values."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = Normal(loc=loc, scale=scale)

        key = jr.key(0)
        sample, log_prob = dist.sample_and_log_prob(key)

        expected_log_prob = dist.log_prob(sample)
        assert jnp.allclose(log_prob, expected_log_prob)

    def test_log_prob_standard_normal(self):
        """Test log probability for standard normal."""
        dist = Normal(loc=jnp.array([0.0]), scale=jnp.array([1.0]))

        log_prob_at_mean = dist.log_prob(jnp.array([0.0]))
        expected = -0.5 * jnp.log(2 * jnp.pi)
        assert jnp.allclose(log_prob_at_mean, expected)

    def test_log_prob_values(self):
        """Test log probability at various points."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = Normal(loc=loc, scale=scale)

        log_prob = dist.log_prob(jnp.array([1.0]))
        expected = -0.5 * jnp.log(2 * jnp.pi) - 0.5
        assert jnp.allclose(log_prob, expected)

    def test_prob(self):
        """Test probability computation."""
        dist = Normal(loc=jnp.array([0.0]), scale=jnp.array([1.0]))

        prob_at_mean = dist.prob(jnp.array([0.0]))
        expected = 1.0 / jnp.sqrt(2 * jnp.pi)
        assert jnp.allclose(prob_at_mean, expected)

    def test_entropy(self):
        """Test entropy computation."""
        scale = jnp.array([2.0])
        dist = Normal(loc=jnp.array([0.0]), scale=scale)

        entropy = dist.entropy()
        expected = 0.5 * jnp.log(2 * jnp.pi * jnp.e * scale**2)
        assert jnp.allclose(entropy, expected)

    def test_entropy_increases_with_scale(self):
        """Test that entropy increases with larger scale."""
        dist1 = Normal(loc=jnp.array([0.0]), scale=jnp.array([1.0]))
        dist2 = Normal(loc=jnp.array([0.0]), scale=jnp.array([2.0]))

        assert dist2.entropy() > dist1.entropy()

    def test_mean(self):
        """Test mean computation."""
        loc = jnp.array([1.0, 2.0, 3.0])
        scale = jnp.array([1.0, 2.0, 3.0])
        dist = Normal(loc=loc, scale=scale)

        mean = dist.mean()
        assert jnp.allclose(mean, loc)

    def test_mode(self):
        """Test mode computation (same as mean for normal)."""
        loc = jnp.array([1.0, 2.0, 3.0])
        scale = jnp.array([1.0, 2.0, 3.0])
        dist = Normal(loc=loc, scale=scale)

        mode = dist.mode()
        assert jnp.allclose(mode, loc)

    def test_samples_follow_distribution(self):
        """Test that samples approximately follow the distribution."""
        loc = jnp.array([5.0])
        scale = jnp.array([2.0])
        dist = Normal(loc=loc, scale=scale)

        keys = jr.split(jr.key(0), 10000)
        samples = jnp.array([dist.sample(k)[0] for k in keys])

        assert jnp.abs(samples.mean() - 5.0) < 0.1
        assert jnp.abs(samples.std() - 2.0) < 0.1

    def test_batched_operations(self):
        """Test that operations work with batched inputs."""
        loc = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        scale = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        dist = Normal(loc=loc, scale=scale)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (2, 2)

    def test_scalar_inputs(self):
        """Test with scalar inputs."""
        dist = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == ()
