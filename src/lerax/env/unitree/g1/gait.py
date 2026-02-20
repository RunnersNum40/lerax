"""Pure-function gait phase utilities for bipedal locomotion.

Provides phase-based gait generation where each foot follows a sinusoidal
pattern offset by pi (alternating stance/swing). Phase advances at a
configurable gait frequency and drives desired foot height trajectories
via cubic bezier interpolation.
"""

from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array, Float


def initial_gait_phase() -> Float[Array, "2"]:
    """Return initial gait phase for left and right feet.

    Left foot starts at 0 (stance), right foot at pi (swing).

    Returns:
        Phase array [left, right] in [-pi, pi).
    """
    return jnp.array([0.0, jnp.pi])


def advance_gait_phase(
    phase: Float[Array, "2"],
    frequency: Float[Array, ""],
    dt: Float[Array, ""],
) -> Float[Array, "2"]:
    """Advance gait phase by one timestep.

    Args:
        phase: Current phase [left, right] in [-pi, pi).
        frequency: Gait frequency in Hz.
        dt: Control timestep in seconds.

    Returns:
        Updated phase wrapped to [-pi, pi).
    """
    phase_increment = 2 * jnp.pi * frequency * dt
    next_phase = phase + phase_increment
    return jnp.fmod(next_phase + jnp.pi, 2 * jnp.pi) - jnp.pi


def gait_phase_observation(phase: Float[Array, "2"]) -> Float[Array, "4"]:
    """Convert gait phase to sin/cos observation.

    Args:
        phase: Phase [left, right].

    Returns:
        [cos(left), sin(left), cos(right), sin(right)].
    """
    return jnp.concatenate([jnp.cos(phase), jnp.sin(phase)])


def desired_foot_height(
    phase: Float[Array, "2"],
    swing_height: float | Float[Array, ""] = 0.15,
) -> Float[Array, "2"]:
    """Compute desired foot z-height from gait phase.

    Uses cubic bezier interpolation following MuJoCo Playground's foot
    trajectory generation. During the first half of the phase the foot rises
    from ground to swing_height; during the second half it descends back.

    Args:
        phase: Phase [left, right] in [-pi, pi).
        swing_height: Maximum foot clearance height in meters.

    Returns:
        Desired foot height for [left, right].
    """

    def _cubic_bezier(
        y_start: float | Float[Array, ""],
        y_end: float | Float[Array, ""],
        x: Float[Array, ""],
    ) -> Float[Array, ""]:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phase + jnp.pi) / (2 * jnp.pi)
    stance = _cubic_bezier(0, swing_height, 2 * x)
    swing = _cubic_bezier(swing_height, 0, 2 * x - 1)
    return jnp.where(x <= 0.5, stance, swing)
