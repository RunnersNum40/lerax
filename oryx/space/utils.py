from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jaxtyping import Array


def try_cast(x: Any) -> Array | None:
    try:
        return jnp.asarray(x)
    except TypeError:
        return None
