from __future__ import annotations

from typing import Sequence

import jax
from distreqx import distributions
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, Key

from .base_distribution import AbstractMaskableDistribution


class MultiCategorical(
    AbstractMaskableDistribution[
        Integer[Array, " dims"],
        Bool[Array, "... sum_of_classes"] | Sequence[Bool[Array, "... classes"]],
    ]
):
    """
    Product of independent Categorical distributions.

    Attributes:
        distribution: Tuple of underlying Categorical distributions.
        action_dims: Tuple of number of classes for each categorical dimension.

    """

    distribution: tuple[distributions.Categorical, ...]
    action_dims: tuple[int, ...]

    def __init__(
        self,
        logits: (
            Float[ArrayLike, " sum_of_classes"]
            | Sequence[Float[ArrayLike, " classes"]]
            | None
        ) = None,
        probs: (
            Float[ArrayLike, " sum_of_classes"]
            | Sequence[Float[ArrayLike, " classes"]]
            | None
        ) = None,
        action_dims: Sequence[int] | None = None,
    ):
        if (logits is None) == (probs is None):
            raise ValueError("Exactly one of logits or probs must be provided.")

        if action_dims is not None:
            action_dims_tup = tuple(int(d) for d in action_dims)
            if len(action_dims_tup) == 0:
                raise ValueError("action_dims must be non-empty.")
            if any(d <= 0 for d in action_dims_tup):
                raise ValueError("All entries in action_dims must be positive.")
        else:
            action_dims_tup = None

        if logits is not None:
            pieces, inferred_dims = self._split_or_unpack_params(
                logits, action_dims_tup
            )
            self.action_dims = inferred_dims
            self.distribution = tuple(
                distributions.Categorical(logits=piece) for piece in pieces
            )
        else:
            assert probs is not None
            pieces, inferred_dims = self._split_or_unpack_params(probs, action_dims_tup)
            self.action_dims = inferred_dims
            self.distribution = tuple(
                distributions.Categorical(probs=piece) for piece in pieces
            )

    @staticmethod
    def _split_or_unpack_params(
        params: (
            Float[ArrayLike, " sum_of_classes"] | Sequence[Float[ArrayLike, " classes"]]
        ),
        action_dims: tuple[int, ...] | None,
    ) -> tuple[tuple[Float[Array, "... classes"], ...], tuple[int, ...]]:
        # Sequence form: [ ..., classes_i ] for each action dim
        if isinstance(params, (list, tuple)):
            if len(params) == 0:
                raise ValueError("Sequence params must be non-empty.")
            arrays = tuple(jnp.asarray(p) for p in params)
            inferred_dims = tuple(int(a.shape[-1]) for a in arrays)

            if any(d <= 0 for d in inferred_dims):
                raise ValueError(
                    "Each categorical dimension must have at least 1 class."
                )

            if action_dims is not None and inferred_dims != action_dims:
                raise ValueError(
                    f"action_dims={action_dims} does not match params last-dims={inferred_dims}."
                )

            lead_shape = arrays[0].shape[:-1]
            for i, a in enumerate(arrays):
                if a.shape[:-1] != lead_shape:
                    raise ValueError(
                        "All parameter pieces must have the same leading (batch) shape. "
                        f"Piece 0 has {lead_shape}, piece {i} has {a.shape[:-1]}."
                    )

            return arrays, inferred_dims

        # Flat form: [..., sum_of_classes] + action_dims required
        arr = jnp.asarray(params)
        if action_dims is None:
            raise ValueError(
                "action_dims must be provided when logits/probs is a single concatenated array."
            )

        total = int(sum(action_dims))
        if arr.shape[-1] != total:
            raise ValueError(
                f"Last dimension ({arr.shape[-1]}) must equal sum(action_dims) ({total})."
            )

        split_idx = jnp.cumsum(jnp.asarray(action_dims[:-1]))
        pieces = tuple(jnp.split(arr, split_idx, axis=-1))
        return pieces, action_dims

    @property
    def logits(self) -> Float[Array, "... sum_of_classes"]:
        return jnp.concatenate(tuple(d.logits for d in self.distribution), axis=-1)

    @property
    def probs(self) -> Float[Array, "... sum_of_classes"]:
        return jnp.concatenate(tuple(d.probs for d in self.distribution), axis=-1)

    def mask(
        self,
        mask: Bool[Array, "... sum_of_classes"] | Sequence[Bool[Array, "... classes"]],
    ) -> MultiCategorical:
        mask_pieces, _ = self._split_or_unpack_params(mask, self.action_dims)
        masked_logits_pieces = tuple(
            jnp.where(m, d.logits, -jnp.inf)
            for d, m in zip(self.distribution, mask_pieces)
        )
        masked_logits = jnp.concatenate(masked_logits_pieces, axis=-1)
        return MultiCategorical(logits=masked_logits, action_dims=self.action_dims)

    def log_prob(self, value: Integer[ArrayLike, " dims"]) -> Float[Array, "..."]:
        value_arr = jnp.asarray(value)
        if value_arr.shape[-1] != len(self.action_dims):
            raise ValueError(
                f"value last dimension ({value_arr.shape[-1]}) must equal number of action dims "
                f"({len(self.action_dims)})."
            )
        logps = tuple(
            d.log_prob(value_arr[..., i]) for i, d in enumerate(self.distribution)
        )
        return jnp.sum(jnp.stack(logps, axis=-1), axis=-1)

    def prob(self, value: Integer[ArrayLike, " dims"]) -> Float[Array, "..."]:
        return jnp.exp(self.log_prob(value))

    def sample(self, key: Key) -> Integer[Array, " ... dims"]:
        keys = jax.random.split(key, len(self.action_dims))
        samples = tuple(d.sample(k) for d, k in zip(self.distribution, keys))
        return jnp.stack(samples, axis=-1)

    def entropy(self) -> Float[Array, "..."]:
        ents = tuple(d.entropy() for d in self.distribution)
        return jnp.sum(jnp.stack(ents, axis=-1), axis=-1)

    def mean(self) -> Float[Array, " ... dims"]:
        means = tuple(d.mean() for d in self.distribution)
        return jnp.stack(means, axis=-1)

    def mode(self) -> Integer[Array, " ... dims"]:
        modes = tuple(d.mode() for d in self.distribution)
        return jnp.stack(modes, axis=-1)

    def sample_and_log_prob(
        self, key: Key
    ) -> tuple[Integer[Array, " ... dims"], Float[Array, "..."]]:
        keys = jax.random.split(key, len(self.action_dims))
        pairs = tuple(d.sample_and_log_prob(k) for d, k in zip(self.distribution, keys))
        samples = jnp.stack(tuple(p[0] for p in pairs), axis=-1)
        logps = jnp.sum(jnp.stack(tuple(p[1] for p in pairs), axis=-1), axis=-1)
        return samples, logps
