import equinox as eqx
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jax import numpy as jnp
from jaxtyping import Array, Float


def check_make_safe_invariants(
    ts: Float[Array, " n"], ys: Float[Array, " n d"]
) -> tuple[Float[Array, " n"], Float[Array, " n d"]]:
    ts_nan = jnp.isnan(ts)
    ys_row_nan = jnp.all(jnp.isnan(ys), axis=-1)

    ts = eqx.error_if(ts, ts_nan != ys_row_nan, "NaN rows in ys must match NaNs in ts.")

    transitions = jnp.sum(ts_nan[:-1] != ts_nan[1:])
    edges_only = (transitions <= 1) | ((transitions == 2) & ts_nan[0] & ts_nan[-1])
    any_valid = jnp.any(~ts_nan)
    ts = eqx.error_if(
        ts,
        ~(edges_only & any_valid),
        "NaNs in ts may only appear at the edges; the non-NaN region must be a single contiguous block with at least one valid timestamp.",
    )

    increasing = jnp.where(ts_nan[1:] | ts_nan[:-1], True, ts[1:] > ts[:-1])
    ts = eqx.error_if(ts, ~increasing, "ts must be strictly increasing where not NaN.")
    return ts, ys


def make_monotonic_increasing(ts: Float[Array, " n"]) -> Float[Array, " n"]:
    """
    Replace edge NaNs while making ts strictly increasing.

    Assumes make_safe_invariants have been checked.
    """
    n = ts.shape[0]
    idxs = jnp.arange(n)
    nan_mask = jnp.isnan(ts)

    first_valid_idx = jnp.argmax(~nan_mask)
    first_val = ts[first_valid_idx]
    step_left = jnp.maximum(jnp.spacing(first_val), jnp.finfo(ts.dtype).eps)
    ts_filled = jnp.where(
        idxs < first_valid_idx,
        first_val - step_left * (first_valid_idx - idxs),
        ts,
    )

    last_valid_idx = n - 1 - jnp.argmax((~nan_mask)[::-1])
    last_val = ts[last_valid_idx]
    step_right = jnp.maximum(jnp.spacing(last_val), jnp.finfo(ts.dtype).eps)
    ts_filled = jnp.where(
        idxs > last_valid_idx,
        last_val + step_right * (idxs - last_valid_idx),
        ts_filled,
    )

    return ts_filled


def pad_ends(xs: Float[Array, " n d"]) -> Float[Array, " n d"]:
    """
    Replace NaN rows at the start and end of xs with the nearest valid row.

    Assumes make_safe_invariants have been checked.
    """
    n = xs.shape[0]
    idxs = jnp.arange(n)
    row_nan = jnp.all(jnp.isnan(xs), axis=-1)

    first_valid_idx = jnp.argmax(~row_nan)
    leading_mask = (idxs < first_valid_idx) & row_nan
    filled = jnp.where(
        leading_mask[:, None],
        jnp.broadcast_to(xs[first_valid_idx], xs.shape),
        xs,
    )

    last_valid_idx = n - 1 - jnp.argmax((~row_nan)[::-1])
    trailing_mask = (idxs > last_valid_idx) & row_nan
    filled = jnp.where(
        trailing_mask[:, None],
        jnp.broadcast_to(xs[last_valid_idx], xs.shape),
        filled,
    )
    return filled


def make_safe(
    ts: Float[Array, " n"], ys: Float[Array, " n d"]
) -> tuple[Float[Array, " n"], Float[Array, " n d"]]:
    ts, ys = check_make_safe_invariants(ts, ys)
    return make_monotonic_increasing(ts), pad_ends(ys)


def safe_control(
    ts: Float[Array, " n"], ys: Float[Array, " n d"]
) -> CubicInterpolation:
    ts, ys = make_safe(ts, ys)
    coeffs = backward_hermite_coefficients(ts, ys)
    return CubicInterpolation(ts, coeffs)
