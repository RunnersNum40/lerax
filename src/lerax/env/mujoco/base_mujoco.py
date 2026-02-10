from __future__ import annotations

import equinox as eqx
import jax

try:
    import mujoco
    from mujoco import mjx
except ImportError as e:
    raise ImportError(
        "MuJoCo environments require the optional mujoco dependency. "
        "Install it with: pip install 'lerax[mujoco]'"
    ) from e

from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Key, PyTree

from lerax.render import AbstractRenderer
from lerax.render.mujoco_renderer import MujocoRenderer
from lerax.space import Box

from ..base_env import AbstractEnv, AbstractEnvState


def fix_dtype(x: Array) -> Array:
    mapping = {
        jnp.floating: jnp.float_,
        jnp.integer: jnp.int_,
        jnp.uint: jnp.uint,
        jnp.bool: jnp.bool_,
    }

    for base_type, target_type in mapping.items():
        if jnp.issubdtype(x.dtype, base_type):
            return x.astype(target_type)

    return x


def tree_fix_dtype(tree: PyTree) -> PyTree:
    return jax.tree.map(fix_dtype, tree)


class MujocoEnvState(AbstractEnvState):
    sim_state: eqx.AbstractVar[mjx.Data] = eqx.field(converter=tree_fix_dtype)
    t: eqx.AbstractVar[Float[Array, ""]]


class AbstractMujocoEnv[
    ActType: Float[Array, "..."],
    ObsType: Float[Array, "..."],
](AbstractEnv[MujocoEnvState, ActType, ObsType, None]):
    name: eqx.AbstractVar[str]

    action_space: eqx.AbstractVar[Box]
    observation_space: eqx.AbstractVar[Box]

    model: eqx.AbstractVar[mjx.Model]
    mujoco_model: eqx.AbstractVar[mujoco.MjModel]
    frame_skip: eqx.AbstractVar[int]
    dt: eqx.AbstractVar[float]

    def action_mask(self, state: MujocoEnvState, *, key: Key[Array, ""]) -> None:
        return None

    def transition(
        self, state: MujocoEnvState, action: ActType, *, key: Key[Array, ""]
    ) -> MujocoEnvState:
        def step_once(data: mjx.Data, _) -> tuple[mjx.Data, None]:
            data = mjx.step(self.model, data)
            return data, None

        data = state.sim_state.replace(ctrl=action)
        data, _ = lax.scan(step_once, data, None, length=self.frame_skip)

        return eqx.tree_at(
            lambda s: (s.sim_state, s.t), state, (data, state.t + self.dt)
        )

    def truncate(self, state: MujocoEnvState) -> Bool[Array, ""]:
        return jnp.array(False)

    def state_info(self, state: MujocoEnvState) -> dict:
        return {}

    def default_renderer(self) -> MujocoRenderer:
        return MujocoRenderer(self.mujoco_model)

    def render(self, state: MujocoEnvState, renderer: AbstractRenderer):
        if not isinstance(renderer, MujocoRenderer):
            raise TypeError("Mujoco environment requires a Mujoco renderer.")

        renderer.render(state.sim_state)
        renderer.draw()
