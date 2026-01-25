import pytest
from jax import numpy as jnp
from jax import random as jr

from lerax.distribution import MultivariateNormalDiag


class TestMultivariateNormalDiag:
    """Tests for the MultivariateNormalDiag distribution."""

    def test_init(self):
        """Test basic initialization."""
        loc = jnp.array([0.0, 1.0, 2.0])
        scale_diag = jnp.array([1.0, 2.0, 3.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        assert jnp.allclose(dist.loc, loc)
        assert jnp.allclose(dist.scale_diag, scale_diag)

    def test_init_with_none_loc(self):
        """Test initialization with default loc."""
        scale_diag = jnp.array([1.0, 2.0])
        dist = MultivariateNormalDiag(scale_diag=scale_diag)

        assert dist.loc.shape == scale_diag.shape

    def test_init_with_none_scale(self):
        """Test initialization with default scale."""
        loc = jnp.array([0.0, 1.0])
        dist = MultivariateNormalDiag(loc=loc)

        assert dist.scale_diag.shape == loc.shape

    def test_init_validates_shape_match(self):
        """Test that loc and scale_diag must have the same shape."""
        with pytest.raises(ValueError):
            MultivariateNormalDiag(
                loc=jnp.array([0.0, 1.0, 2.0]), scale_diag=jnp.array([1.0, 2.0])
            )

    def test_sample_shape(self):
        """Test that samples have the correct shape."""
        loc = jnp.array([0.0, 1.0, 2.0])
        scale_diag = jnp.array([1.0, 1.0, 1.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (3,)
        assert jnp.issubdtype(sample.dtype, jnp.floating)

    def test_sample_and_log_prob(self):
        """Test sample_and_log_prob returns consistent values."""
        loc = jnp.array([0.0, 0.0])
        scale_diag = jnp.array([1.0, 1.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        key = jr.key(0)
        sample, log_prob = dist.sample_and_log_prob(key)

        expected_log_prob = dist.log_prob(sample)
        assert jnp.allclose(log_prob, expected_log_prob)

    def test_log_prob_at_mean(self):
        """Test log probability at the mean."""
        dims = 3
        loc = jnp.zeros(dims)
        scale_diag = jnp.ones(dims)
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        log_prob_at_mean = dist.log_prob(loc)
        expected = -0.5 * dims * jnp.log(2 * jnp.pi)
        assert jnp.allclose(log_prob_at_mean, expected)

    def test_prob(self):
        """Test probability computation."""
        loc = jnp.array([0.0, 0.0])
        scale_diag = jnp.array([1.0, 1.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        prob_at_mean = dist.prob(loc)
        expected = (2 * jnp.pi) ** (-1)
        assert jnp.allclose(prob_at_mean, expected)

    def test_entropy(self):
        """Test entropy computation."""
        dims = 2
        scale_diag = jnp.array([2.0, 3.0])
        dist = MultivariateNormalDiag(loc=jnp.zeros(dims), scale_diag=scale_diag)

        entropy = dist.entropy()
        expected = 0.5 * dims * (1 + jnp.log(2 * jnp.pi)) + jnp.sum(jnp.log(scale_diag))
        assert jnp.allclose(entropy, expected)

    def test_entropy_independent_of_loc(self):
        """Test that entropy is independent of location."""
        scale_diag = jnp.array([1.0, 2.0])
        dist1 = MultivariateNormalDiag(loc=jnp.zeros(2), scale_diag=scale_diag)
        dist2 = MultivariateNormalDiag(
            loc=jnp.array([100.0, -100.0]), scale_diag=scale_diag
        )

        assert jnp.allclose(dist1.entropy(), dist2.entropy())

    def test_mean(self):
        """Test mean computation."""
        loc = jnp.array([1.0, 2.0, 3.0])
        scale_diag = jnp.array([1.0, 2.0, 3.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        mean = dist.mean()
        assert jnp.allclose(mean, loc)

    def test_mode(self):
        """Test mode computation (same as mean for MVN)."""
        loc = jnp.array([1.0, 2.0, 3.0])
        scale_diag = jnp.array([1.0, 2.0, 3.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        mode = dist.mode()
        assert jnp.allclose(mode, loc)

    def test_samples_independent_dimensions(self):
        """Test that samples have independent dimensions."""
        loc = jnp.zeros(2)
        scale_diag = jnp.array([1.0, 1.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        keys = jr.split(jr.key(0), 5000)
        samples = jnp.array([dist.sample(k) for k in keys])

        corr = jnp.corrcoef(samples.T)[0, 1]
        assert jnp.abs(corr) < 0.05

    def test_higher_dimensional_loc(self):
        """Test with higher dimensional loc/scale."""
        loc = jnp.array([0.0, 1.0, 2.0])
        scale_diag = jnp.array([1.0, 1.0, 1.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        key = jr.key(0)
        sample = dist.sample(key)

        assert sample.shape == (3,)

    def test_log_prob_multiple_values(self):
        """Test log_prob with multiple evaluation points using vmap."""
        import jax

        loc = jnp.array([0.0, 0.0])
        scale_diag = jnp.array([1.0, 1.0])
        dist = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        values = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        log_probs = jax.vmap(dist.log_prob)(values)

        assert log_probs.shape == (2,)
