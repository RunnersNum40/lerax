import jax.numpy as jnp
import pytest
from jax import random as jr

from oryx.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple


def test_discrete_contains_and_repr():
    s = Discrete(5, start=2)
    assert s.contains(2)
    assert s.contains(6 - 1)
    assert not s.contains(1)
    assert repr(s) == "Discrete(5, start=2)"


def test_discrete_sample_bounds():
    key = jr.key(0)
    s = Discrete(3, start=1)
    x = s.sample(key)
    assert 1 <= int(x) < 4


@pytest.mark.parametrize(
    "low, high, shape",
    [
        (0.0, 1.0, None),
        ([-1, 0], [1, 2], (2,)),
    ],
)
def test_box_contains_and_repr(low, high, shape):
    key = jr.key(1)
    box = Box(low, high, shape)
    sample = box.sample(key)
    assert jnp.all(sample >= box.low)
    assert jnp.all(sample <= box.high)
    mid = (box.low + box.high) / 2
    assert box.contains(mid)
    assert not box.contains(box.low - 1.0)
    assert repr(box) == f"Box(low={box.low}, high={box.high})"


def test_tuple_space():
    key = jr.key(2)
    t = Tuple((Discrete(2), Box(0, 1, shape=())))
    sample = t.sample(key)
    assert isinstance(sample, tuple)
    assert t.contains(sample)
    assert not t.contains((10, 0.5))
    assert repr(t).startswith("Tuple(")


def test_dict_space():
    key = jr.key(3)
    spaces = {"a": Discrete(2), "b": Box(0, 1, shape=())}
    d = Dict(spaces)
    sample = d.sample(key)
    assert set(sample.keys()) == set(spaces.keys())
    assert d.contains(sample)
    assert not d.contains({"a": -1, "b": 0.5})
    assert "Dict(" in repr(d)


def test_multidiscrete():
    key = jr.key(4)
    md = MultiDiscrete((2, 3), starts=(1, 2))
    x = md.sample(key)
    assert x.shape == (2,)
    assert md.contains(x)
    assert not md.contains(jnp.array([0, 5]))
    assert repr(md).startswith("MultiDiscrete")
