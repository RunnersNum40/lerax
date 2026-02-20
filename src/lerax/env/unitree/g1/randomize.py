"""Domain randomization functions for sim-to-real transfer.

All functions are pure and operate on ``mjx.Model`` instances. The randomized
model is stored on the environment state so that each episode can have different
dynamics while remaining fully JIT-compatible.

Randomization ranges follow MuJoCo Playground conventions.
"""

from __future__ import annotations

from jax import random as jr
from jaxtyping import Array, Float, Key
from mujoco import mjx


def randomize_friction(
    model: mjx.Model,
    *,
    key: Key[Array, ""],
    friction_range: tuple[float, float] = (0.4, 1.0),
) -> mjx.Model:
    """Randomize floor/foot contact friction.

    Applies a uniform random friction coefficient to the first two
    contact pairs (left_foot_floor, right_foot_floor).

    Args:
        model: MJX model to randomize.
        key: PRNG key.
        friction_range: Min and max friction coefficient.

    Returns:
        Model with randomized pair friction.
    """
    friction = jr.uniform(key, minval=friction_range[0], maxval=friction_range[1])
    pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)
    return model.tree_replace({"pair_friction": pair_friction})


def randomize_friction_loss(
    model: mjx.Model,
    *,
    key: Key[Array, ""],
    nominal_friction_loss: Float[Array, "..."],
    scale_range: tuple[float, float] = (0.5, 2.0),
) -> mjx.Model:
    """Randomize joint friction loss for actuated DOFs.

    Scales the friction loss of actuated joints (indices 6+, skipping
    the 6 free-joint DOFs).

    Args:
        model: MJX model to randomize.
        key: PRNG key.
        nominal_friction_loss: Original friction loss values for actuated DOFs.
        scale_range: Min and max scale factors.

    Returns:
        Model with randomized dof_frictionloss.
    """
    num_actuated = nominal_friction_loss.shape[0]
    scales = jr.uniform(
        key, shape=(num_actuated,), minval=scale_range[0], maxval=scale_range[1]
    )
    frictionloss = nominal_friction_loss * scales
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)
    return model.tree_replace({"dof_frictionloss": dof_frictionloss})


def randomize_armature(
    model: mjx.Model,
    *,
    key: Key[Array, ""],
    nominal_armature: Float[Array, "..."],
    scale_range: tuple[float, float] = (1.0, 1.05),
) -> mjx.Model:
    """Randomize joint armature for actuated DOFs.

    Args:
        model: MJX model to randomize.
        key: PRNG key.
        nominal_armature: Original armature values for actuated DOFs.
        scale_range: Min and max scale factors.

    Returns:
        Model with randomized dof_armature.
    """
    num_actuated = nominal_armature.shape[0]
    scales = jr.uniform(
        key, shape=(num_actuated,), minval=scale_range[0], maxval=scale_range[1]
    )
    armature = nominal_armature * scales
    dof_armature = model.dof_armature.at[6:].set(armature)
    return model.tree_replace({"dof_armature": dof_armature})


def randomize_body_mass(
    model: mjx.Model,
    *,
    key: Key[Array, ""],
    nominal_body_mass: Float[Array, "..."],
    scale_range: tuple[float, float] = (0.9, 1.1),
    torso_body_id: int,
    torso_offset_range: tuple[float, float] = (-1.0, 1.0),
) -> mjx.Model:
    """Randomize body link masses with extra torso offset.

    Scales all body masses uniformly, then adds an additional offset
    to the torso mass to simulate payload variation.

    Args:
        model: MJX model to randomize.
        key: PRNG key.
        nominal_body_mass: Original body masses.
        scale_range: Min and max mass scale factors.
        torso_body_id: Body index for the torso.
        torso_offset_range: Min and max additional torso mass offset (kg).

    Returns:
        Model with randomized body_mass.
    """
    mass_key, torso_key = jr.split(key)
    scales = jr.uniform(
        mass_key, shape=(model.nbody,), minval=scale_range[0], maxval=scale_range[1]
    )
    body_mass = nominal_body_mass * scales
    torso_offset = jr.uniform(
        torso_key, minval=torso_offset_range[0], maxval=torso_offset_range[1]
    )
    body_mass = body_mass.at[torso_body_id].set(body_mass[torso_body_id] + torso_offset)
    return model.tree_replace({"body_mass": body_mass})


def randomize_model(
    model: mjx.Model,
    *,
    key: Key[Array, ""],
    nominal_friction_loss: Float[Array, "..."],
    nominal_armature: Float[Array, "..."],
    nominal_body_mass: Float[Array, "..."],
    torso_body_id: int,
    friction_range: tuple[float, float] = (0.4, 1.0),
    friction_loss_scale_range: tuple[float, float] = (0.5, 2.0),
    armature_scale_range: tuple[float, float] = (1.0, 1.05),
    mass_scale_range: tuple[float, float] = (0.9, 1.1),
    torso_offset_range: tuple[float, float] = (-1.0, 1.0),
) -> mjx.Model:
    """Apply all domain randomizations to a model.

    Args:
        model: MJX model to randomize.
        key: PRNG key.
        nominal_friction_loss: Original actuated DOF friction loss.
        nominal_armature: Original actuated DOF armature.
        nominal_body_mass: Original body masses.
        torso_body_id: Body index for the torso.
        friction_range: Friction coefficient range.
        friction_loss_scale_range: Friction loss scale range.
        armature_scale_range: Armature scale range.
        mass_scale_range: Body mass scale range.
        torso_offset_range: Torso mass offset range.

    Returns:
        Fully randomized model.
    """
    friction_key, floss_key, armature_key, mass_key = jr.split(key, 4)

    model = randomize_friction(model, key=friction_key, friction_range=friction_range)
    model = randomize_friction_loss(
        model,
        key=floss_key,
        nominal_friction_loss=nominal_friction_loss,
        scale_range=friction_loss_scale_range,
    )
    model = randomize_armature(
        model,
        key=armature_key,
        nominal_armature=nominal_armature,
        scale_range=armature_scale_range,
    )
    model = randomize_body_mass(
        model,
        key=mass_key,
        nominal_body_mass=nominal_body_mass,
        scale_range=mass_scale_range,
        torso_body_id=torso_body_id,
        torso_offset_range=torso_offset_range,
    )
    return model


def add_action_noise(
    action: Float[Array, "..."],
    *,
    key: Key[Array, ""],
    scale: float = 0.0,
) -> Float[Array, "..."]:
    """Add uniform noise to actions.

    Args:
        action: Action array.
        key: PRNG key.
        scale: Noise amplitude (uniform in [-scale, scale]).

    Returns:
        Noisy action.
    """
    noise = (2 * jr.uniform(key, shape=action.shape) - 1) * scale
    return action + noise


def add_observation_noise(
    observation: Float[Array, "..."],
    *,
    key: Key[Array, ""],
    noise_level: float = 1.0,
    noise_scales: dict[str, float],
    component_slices: dict[str, tuple[int, int]],
) -> Float[Array, "..."]:
    """Add per-component uniform noise to a flat observation vector.

    Args:
        observation: Flat observation array.
        key: PRNG key.
        noise_level: Global noise multiplier (0 disables noise).
        noise_scales: Per-component noise scale.
        component_slices: Mapping from component name to (start, end) indices.

    Returns:
        Observation with per-component noise added.
    """
    noisy = observation
    for component_name, (start, end) in component_slices.items():
        if component_name in noise_scales:
            key, subkey = jr.split(key)
            component_noise = (
                (2 * jr.uniform(subkey, shape=(end - start,)) - 1)
                * noise_level
                * noise_scales[component_name]
            )
            noisy = noisy.at[start:end].add(component_noise)
    return noisy
