import equinox as eqx
import jax.numpy as jnp

from lerax.target import SoftUpdate
from lerax.utils import polyak_average


class Params(eqx.Module):
    weight: jnp.ndarray
    label: str


def test_soft_update_tau_one_copies_online_exactly():
    online = Params(weight=jnp.array([1.0, 2.0, 3.0]), label="online")
    target = Params(weight=jnp.array([-1.0, -2.0, -3.0]), label="target")

    updated = SoftUpdate(tau=1.0)(online, target)

    assert jnp.allclose(updated.weight, online.weight)


def test_soft_update_tau_zero_keeps_target_unchanged():
    online = Params(weight=jnp.array([1.0, 2.0, 3.0]), label="online")
    target = Params(weight=jnp.array([-1.0, -2.0, -3.0]), label="target")

    updated = SoftUpdate(tau=0.0)(online, target)

    assert jnp.allclose(updated.weight, target.weight)


def test_soft_update_interpolates_at_tau_half():
    online = Params(weight=jnp.array([2.0, 4.0]), label="online")
    target = Params(weight=jnp.array([0.0, 0.0]), label="target")

    updated = SoftUpdate(tau=0.5)(online, target)

    assert jnp.allclose(updated.weight, jnp.array([1.0, 2.0]))


def test_soft_update_takes_non_array_leaves_from_online():
    online = Params(weight=jnp.zeros(2), label="online")
    target = Params(weight=jnp.ones(2), label="target")

    updated = SoftUpdate(tau=0.3)(online, target)

    assert updated.label == "online"


def test_soft_update_matches_polyak_average():
    online = Params(weight=jnp.array([1.0, -2.0]), label="online")
    target = Params(weight=jnp.array([3.0, 4.0]), label="target")

    expected = polyak_average(online, target, 0.2)
    actual = SoftUpdate(tau=0.2)(online, target)

    assert jnp.allclose(actual.weight, expected.weight)
