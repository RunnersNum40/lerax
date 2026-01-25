import pytest
from jax import numpy as jnp
from jax import random as jr

from lerax.distribution import SquashedNormal


class TestSquashedNormal:
    """Tests for the SquashedNormal distribution."""

    def test_init_default_bounds(self):
        """Test initialization with default bounds [-1, 1]."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        assert jnp.allclose(dist.loc, loc)
        assert jnp.allclose(dist.scale, scale)

    def test_init_custom_bounds(self):
        """Test initialization with custom bounds."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        high = jnp.array(2.0)
        low = jnp.array(-2.0)
        dist = SquashedNormal(loc=loc, scale=scale, high=high, low=low)

        assert jnp.allclose(dist.loc, loc)
        assert jnp.allclose(dist.scale, scale)

    def test_sample_shape(self):
        """Test that samples have the correct shape."""
        loc = jnp.array([0.0, 1.0, 2.0])
        scale = jnp.array([1.0, 1.0, 1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (3,)

    def test_sample_within_bounds_default(self):
        """Test that samples are within default bounds [-1, 1]."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        keys = jr.split(jr.key(0), 1000)
        for k in keys:
            sample = dist.sample(k)
            assert jnp.all(sample >= -1.0)
            assert jnp.all(sample <= 1.0)

    def test_sample_within_bounds_custom(self):
        """Test that samples are within custom bounds."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        high = jnp.array(5.0)
        low = jnp.array(-3.0)
        dist = SquashedNormal(loc=loc, scale=scale, high=high, low=low)

        keys = jr.split(jr.key(0), 1000)
        for k in keys:
            sample = dist.sample(k)
            assert jnp.all(sample >= -3.0)
            assert jnp.all(sample <= 5.0)

    def test_sample_and_log_prob(self):
        """Test sample_and_log_prob returns consistent values."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        key = jr.key(0)
        sample, log_prob = dist.sample_and_log_prob(key)

        expected_log_prob = dist.log_prob(sample)
        assert jnp.allclose(log_prob, expected_log_prob)

    def test_log_prob_finite(self):
        """Test that log_prob is finite for samples."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        keys = jr.split(jr.key(0), 100)
        for k in keys:
            sample = dist.sample(k)
            log_prob = dist.log_prob(sample)
            assert jnp.isfinite(log_prob)

    def test_prob(self):
        """Test probability computation is positive."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        key = jr.key(0)
        sample = dist.sample(key)
        prob = dist.prob(sample)

        assert prob > 0

    def test_entropy_not_implemented(self):
        """Test that entropy raises NotImplementedError for transformed distributions."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        with pytest.raises(NotImplementedError):
            dist.entropy()

    def test_mean_not_implemented(self):
        """Test that mean raises NotImplementedError for non-constant Jacobian."""
        loc = jnp.array([0.0])
        scale = jnp.array([0.1])
        dist = SquashedNormal(loc=loc, scale=scale)

        with pytest.raises(NotImplementedError):
            dist.mean()

    def test_mode(self):
        """Test mode computation."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        mode = dist.mode()
        assert jnp.all(mode >= -1.0)
        assert jnp.all(mode <= 1.0)

    def test_mode_at_center_for_zero_loc(self):
        """Test that mode is at center of bounds for loc=0."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        high = jnp.array(1.0)
        low = jnp.array(-1.0)
        dist = SquashedNormal(loc=loc, scale=scale, high=high, low=low)

        mode = dist.mode()
        assert jnp.allclose(mode, 0.0, atol=0.01)

    def test_bijector_property(self):
        """Test that bijector is accessible."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        dist = SquashedNormal(loc=loc, scale=scale)

        bijector = dist.bijector
        assert bijector is not None

    def test_batched_operations(self):
        """Test that operations work with batched inputs."""
        loc = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        scale = jnp.array([[1.0, 1.0], [0.5, 0.5]])
        dist = SquashedNormal(loc=loc, scale=scale)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (2, 2)
        assert jnp.all(sample >= -1.0)
        assert jnp.all(sample <= 1.0)

    def test_asymmetric_bounds(self):
        """Test with asymmetric bounds."""
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        high = jnp.array(10.0)
        low = jnp.array(0.0)
        dist = SquashedNormal(loc=loc, scale=scale, high=high, low=low)

        keys = jr.split(jr.key(0), 100)
        for k in keys:
            sample = dist.sample(k)
            assert jnp.all(sample >= 0.0)
            assert jnp.all(sample <= 10.0)
