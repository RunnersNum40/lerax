from __future__ import annotations

import equinox as eqx
import pytest
from distreqx import bijectors
from jax import numpy as jnp
from jax import random as jr

from oryx.distribution import (
    Bernoulli,
    Categorical,
    MultivariateNormalDiag,
    Normal,
    SquashedMultivariateNormalDiag,
    SquashedNormal,
)


class TestBernoulli:
    def test_interface(self):
        key = jr.key(0)
        p = jnp.array([0.2, 0.5, 0.8])
        d = Bernoulli(probs=p)

        assert d.logits.shape == p.shape
        assert jnp.array_equal(d.probs, p)

        x = d.sample(key)
        assert x.shape == p.shape
        assert jnp.all(jnp.logical_or(x == 0, x == 1))

        lp = d.log_prob(x)
        assert lp.shape == p.shape
        expected_lp = x * jnp.log(p) + (1 - x) * jnp.log(1 - p)
        assert jnp.allclose(lp, expected_lp, atol=1e-6)

        entropy = -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))
        assert jnp.allclose(d.entropy(), entropy, atol=1e-6)

    def test_jit(self):
        p = jnp.array([0.1, 0.9])
        d = Bernoulli(probs=p)

        @eqx.filter_jit
        def f(dist, key):
            x, lp = dist.sample_and_log_prob(key)
            return x, lp, dist.log_prob(x)

        x, lp1, lp2 = f(d, jr.key(1))
        assert x.shape == p.shape
        assert jnp.allclose(lp1, lp2, atol=1e-6)


class TestCategorical:
    def test_interface(self):
        key = jr.key(0)
        logits = jnp.zeros(4)
        d = Categorical(logits=logits)

        assert d.logits.shape == (4,)
        assert jnp.allclose(d.probs, jnp.full((4,), 0.25))

        x = d.sample(key)
        assert int(x) in range(4)
        assert jnp.allclose(d.probs, jnp.full((4,), 0.25))

        lp = d.log_prob(x)
        assert lp.shape == ()
        assert jnp.allclose(lp, jnp.log(0.25))

        assert jnp.allclose(d.entropy(), jnp.log(4.0))

    def test_jit(self):
        d = Categorical(logits=jnp.array([0.0, 1.0, 0.0]))

        @eqx.filter_jit
        def f(dist, key):
            x = dist.sample(key)
            return dist.log_prob(x)

        out = f(d, jr.key(2))
        assert out.shape == ()


class TestNormal:
    def test_interface(self):
        key = jr.key(0)
        loc = jnp.array([0.0, 1.0])
        scale = jnp.array([1.0, 2.0])
        d = Normal(loc=loc, scale=scale)

        assert jnp.array_equal(d.loc, loc)
        assert jnp.array_equal(d.scale, scale)

        x = d.sample(key)
        assert x.shape == loc.shape
        assert jnp.allclose(d.mean(), loc)
        assert jnp.all(d.log_prob(loc) > d.log_prob(loc + 5.0 * scale))
        assert jnp.allclose(jnp.exp(d.log_prob(loc)), d.prob(loc))

        expected_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.square(scale))
        assert jnp.allclose(d.entropy(), expected_entropy, atol=1e-6)

        with pytest.raises(ValueError):
            Normal(loc=loc, scale=jnp.array(0.0))

    def test_jit(self):
        d = Normal(loc=jnp.array([0.0, 0.5]), scale=jnp.array([1.0, 0.2]))

        @eqx.filter_jit
        def f(dist, key):
            x, lp = dist.sample_and_log_prob(key)
            return lp, dist.log_prob(x)

        lp1, lp2 = f(d, jr.key(3))
        assert jnp.allclose(lp1, lp2, atol=1e-6)


class TestSquashedNormal:
    def test_interface(self):
        key = jr.key(0)
        loc = jnp.array([0.0, 0.5, -0.5])
        scale = jnp.array([0.3, 0.2, 0.4])
        low = jnp.array([0.0, 0.0, -1.0])
        high = jnp.array([1.0, 1.0, 1.0])
        d = SquashedNormal(loc=loc, scale=scale, low=low, high=high)

        assert jnp.array_equal(d.loc, loc)
        assert jnp.array_equal(d.scale, scale)

        x = d.sample(key)
        assert x.shape == loc.shape
        assert jnp.all(x > low) and jnp.all(x < high)
        assert isinstance(d.bijector, bijectors.Chain)

        x2, lp2 = d.sample_and_log_prob(key)
        assert jnp.allclose(d.log_prob(x2), lp2, atol=1e-6)

        with pytest.raises(NotImplementedError):
            d.entropy()

        with pytest.raises(ValueError):
            SquashedNormal(loc=loc, scale=jnp.array(0))

        with pytest.raises(AssertionError):
            SquashedNormal(loc=loc, scale=scale, low=low)

    def test_jit(self):
        d = SquashedNormal(loc=jnp.array([0.0, 0.1]), scale=jnp.array([0.5, 0.4]))

        @eqx.filter_jit
        def f(dist, key):
            x, lp = dist.sample_and_log_prob(key)
            return x, lp, dist.log_prob(x)

        x, lp1, lp2 = f(d, jr.key(4))
        assert jnp.all(x > -1.0) and jnp.all(x < 1.0)
        assert jnp.allclose(lp1, lp2, atol=1e-6)


class TestMultivariateNormalDiag:
    def test_interface(self):
        key = jr.key(0)
        loc = jnp.array([0.0, 1.0, -1.0])
        scale = jnp.array([0.5, 0.3, 0.7])
        d = MultivariateNormalDiag(loc=loc, scale_diag=scale)

        assert jnp.array_equal(d.loc, loc)
        assert jnp.array_equal(d.scale_diag, scale)

        x = d.sample(key)
        assert x.shape == loc.shape
        assert jnp.allclose(d.mean(), loc)

        x2, lp2 = d.sample_and_log_prob(key)
        assert x2.shape == loc.shape and lp2.shape == ()
        assert jnp.allclose(d.log_prob(x2), lp2, atol=1e-6)

        assert d.log_prob(loc) > d.log_prob(loc + 5.0 * scale)

        k = loc.size
        expected_entropy = 0.5 * k * (1.0 + jnp.log(2 * jnp.pi)) + jnp.sum(
            jnp.log(scale)
        )
        assert jnp.allclose(d.entropy(), expected_entropy, atol=1e-6)

        with pytest.raises(ValueError):
            MultivariateNormalDiag(loc=loc, scale_diag=jnp.array(0.0))

    def test_jit(self):
        d = MultivariateNormalDiag(
            loc=jnp.array([0.0, 0.0]),
            scale_diag=jnp.array([0.2, 0.3]),
        )

        @eqx.filter_jit
        def f(dist, key):
            x = dist.sample(key)
            return dist.log_prob(x)

        out = f(d, jr.key(5))
        assert out.shape == ()


class TestSquashedMultivariateNormalDiag:
    def test_tanh_interface(self):
        key = jr.key(0)
        loc = jnp.array([0.1, -0.2])
        scale = jnp.array([0.2, 0.3])
        d = SquashedMultivariateNormalDiag(loc=loc, scale_diag=scale)

        assert jnp.array_equal(d.loc, loc)
        assert jnp.array_equal(d.scale_diag, scale)

        x = d.sample(key)
        assert jnp.all(x > -1.0) and jnp.all(x < 1.0)
        assert isinstance(d.bijector, bijectors.Tanh)

        x2, lp2 = d.sample_and_log_prob(key)
        assert jnp.allclose(d.log_prob(x2), lp2, atol=1e-6)

        mode = d.mode()
        assert jnp.allclose(mode, jnp.tanh(loc), atol=1e-6)

        with pytest.raises(NotImplementedError):
            d.entropy()

    def test_bounded_interface(self):
        key = jr.key(1)
        loc = jnp.array([0.0, 0.5])
        scale = jnp.array([0.2, 0.3])
        low = jnp.array([-2.0, 0.0])
        high = jnp.array([2.0, 3.0])
        d = SquashedMultivariateNormalDiag(
            loc=loc, scale_diag=scale, low=low, high=high
        )

        x = d.sample(key)
        assert jnp.all(x > low) and jnp.all(x < high)
        assert isinstance(d.bijector, bijectors.Chain)

        x2, lp2 = d.sample_and_log_prob(key)
        assert jnp.allclose(d.log_prob(x2), lp2, atol=1e-4)

        with pytest.raises(NotImplementedError):
            d.entropy()

    def test_jit(self):
        tanh = SquashedMultivariateNormalDiag(
            loc=jnp.array([0.0, 0.1]), scale_diag=jnp.array([0.2, 0.3])
        )
        bounded = SquashedMultivariateNormalDiag(
            loc=jnp.array([0.0, 0.1]),
            scale_diag=jnp.array([0.2, 0.3]),
            low=jnp.array([-1.0, 0.0]),
            high=jnp.array([1.0, 2.0]),
        )

        @eqx.filter_jit
        def f(dist, key):
            x, lp = dist.sample_and_log_prob(key)
            return x, lp, dist.log_prob(x)

        for i, dist in enumerate([tanh, bounded], start=6):
            x, lp1, lp2 = f(dist, jr.key(i))
            assert jnp.allclose(lp1, lp2, atol=1e-5)
