import equinox as eqx
import jax.numpy as jnp
import pytest

from lerax.target import HardUpdate, SoftUpdate


class Params(eqx.Module):
    weight: jnp.ndarray


@pytest.mark.parametrize("target_update", [HardUpdate(), SoftUpdate()])
def test_common_target_updates_are_jittable(target_update):
    online = Params(weight=jnp.ones(3))
    target = Params(weight=jnp.zeros(3))

    updated = eqx.filter_jit(target_update)(online, target)

    assert updated.weight.shape == (3,)
    assert jnp.all(jnp.isfinite(updated.weight))


def test_concrete_target_updates_do_not_inherit_from_each_other():
    target_update_types = (HardUpdate, SoftUpdate)

    for target_update_type in target_update_types:
        for other_type in target_update_types:
            if target_update_type is not other_type:
                assert not issubclass(target_update_type, other_type)
