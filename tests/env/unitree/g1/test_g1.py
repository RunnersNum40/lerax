"""Tests for Unitree G1 environments."""

from __future__ import annotations

import jax
from jax import numpy as jnp
from jax import random as jr

from lerax.env.unitree.g1 import G1Locomotion, G1Standing, G1Standup
from lerax.env.unitree.g1.gait import (
    advance_gait_phase,
    desired_foot_height,
    gait_phase_observation,
    initial_gait_phase,
)
from lerax.env.unitree.g1.randomize import (
    randomize_armature,
    randomize_body_mass,
    randomize_friction,
    randomize_friction_loss,
    randomize_model,
)


def test_initial_gait_phase_shape():
    phase = initial_gait_phase()
    assert phase.shape == (2,)
    assert jnp.allclose(phase, jnp.array([0.0, jnp.pi]))


def test_advance_gait_phase_wraps():
    phase = jnp.array([jnp.pi - 0.01, jnp.pi - 0.01])
    next_phase = advance_gait_phase(phase, jnp.array(10.0), jnp.array(0.02))
    assert next_phase.shape == (2,)
    assert jnp.all(next_phase >= -jnp.pi)
    assert jnp.all(next_phase < jnp.pi)


def test_gait_phase_observation_shape():
    phase = initial_gait_phase()
    obs = gait_phase_observation(phase)
    assert obs.shape == (4,)


def test_desired_foot_height_range():
    phase = jnp.linspace(-jnp.pi, jnp.pi, 100)
    phase = jnp.stack([phase, phase], axis=-1)
    heights = jax.vmap(desired_foot_height)(phase)
    assert jnp.all(heights >= -0.01)
    assert jnp.all(heights <= 0.16)


def test_randomize_friction_changes_model():
    env = G1Locomotion()
    model = env.base_model
    original_friction = model.pair_friction[0, 0]
    randomized = randomize_friction(model, key=jr.key(42))
    assert not jnp.allclose(randomized.pair_friction[0, 0], original_friction)


def test_randomize_friction_loss_changes_model():
    env = G1Locomotion()
    model = env.base_model
    original = model.dof_frictionloss[6:]
    randomized = randomize_friction_loss(
        model, key=jr.key(42), nominal_friction_loss=env.nominal_friction_loss
    )
    assert not jnp.allclose(randomized.dof_frictionloss[6:], original)


def test_randomize_armature_changes_model():
    env = G1Locomotion()
    model = env.base_model
    original = model.dof_armature[6:]
    randomized = randomize_armature(
        model, key=jr.key(42), nominal_armature=env.nominal_armature
    )
    assert not jnp.allclose(randomized.dof_armature[6:], original)


def test_randomize_body_mass_changes_model():
    env = G1Locomotion()
    model = env.base_model
    original = model.body_mass.copy()
    randomized = randomize_body_mass(
        model,
        key=jr.key(42),
        nominal_body_mass=env.nominal_body_mass,
        torso_body_id=env.torso_body_id,
    )
    assert not jnp.allclose(randomized.body_mass, original)


def test_randomize_model_comprehensive():
    env = G1Locomotion()
    model = env.base_model
    randomized = randomize_model(
        model,
        key=jr.key(42),
        nominal_friction_loss=env.nominal_friction_loss,
        nominal_armature=env.nominal_armature,
        nominal_body_mass=env.nominal_body_mass,
        torso_body_id=env.torso_body_id,
    )
    assert not jnp.allclose(randomized.pair_friction, model.pair_friction)
    assert not jnp.allclose(randomized.body_mass, model.body_mass)


def test_locomotion_construction():
    env = G1Locomotion()
    assert env.action_space.shape == (29,)
    assert env.observation_space.shape == (103,)


def test_locomotion_frame_skip():
    env = G1Locomotion(control_frequency_hz=50.0)
    assert env.frame_skip == 5
    assert jnp.isclose(env.dt, jnp.array(0.02))


def test_locomotion_frame_skip_25hz():
    env = G1Locomotion(control_frequency_hz=25.0)
    assert env.frame_skip == 10
    assert jnp.isclose(env.dt, jnp.array(0.04))


def test_locomotion_initial():
    env = G1Locomotion()
    state = env.initial(key=jr.key(0))
    assert state.sim_state is not None
    assert state.last_action.shape == (29,)
    assert state.gait_phase.shape == (2,)
    assert state.command.shape == (3,)


def test_locomotion_observation_shape():
    env = G1Locomotion()
    state = env.initial(key=jr.key(0))
    obs = env.observation(state, key=jr.key(1))
    assert obs.shape == (103,)


def test_locomotion_step():
    env = G1Locomotion()
    state = env.initial(key=jr.key(0))
    action = env.action_space.sample(key=jr.key(1))
    next_state = env.transition(state, action, key=jr.key(2))
    reward = env.reward(state, action, next_state, key=jr.key(3))
    terminal = env.terminal(next_state, key=jr.key(4))
    assert reward.shape == ()
    assert terminal.shape == ()


def test_standing_construction():
    env = G1Standing()
    assert env.action_space.shape == (29,)
    assert env.observation_space.shape == (99,)


def test_standing_initial():
    env = G1Standing()
    state = env.initial(key=jr.key(0))
    assert jnp.allclose(state.command, jnp.zeros(3))


def test_standing_observation_shape():
    env = G1Standing()
    state = env.initial(key=jr.key(0))
    obs = env.observation(state, key=jr.key(1))
    assert obs.shape == (99,)


def test_standing_step():
    env = G1Standing()
    state = env.initial(key=jr.key(0))
    action = env.action_space.sample(key=jr.key(1))
    next_state = env.transition(state, action, key=jr.key(2))
    reward = env.reward(state, action, next_state, key=jr.key(3))
    terminal = env.terminal(next_state, key=jr.key(4))
    assert reward.shape == ()
    assert terminal.shape == ()


def test_standup_construction():
    env = G1Standup()
    assert env.action_space.shape == (29,)
    assert env.observation_space.shape == (99,)


def test_standup_initial():
    env = G1Standup()
    state = env.initial(key=jr.key(0))
    assert jnp.allclose(state.command, jnp.zeros(3))


def test_standup_never_terminates():
    env = G1Standup()
    state = env.initial(key=jr.key(0))
    assert not env.terminal(state, key=jr.key(1))


def test_standup_observation_shape():
    env = G1Standup()
    state = env.initial(key=jr.key(0))
    obs = env.observation(state, key=jr.key(1))
    assert obs.shape == (99,)


def test_standup_step():
    env = G1Standup()
    state = env.initial(key=jr.key(0))
    action = env.action_space.sample(key=jr.key(1))
    next_state = env.transition(state, action, key=jr.key(2))
    reward = env.reward(state, action, next_state, key=jr.key(3))
    assert reward.shape == ()
