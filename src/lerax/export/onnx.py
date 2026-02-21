"""
ONNX export for Lerax policies.

Provides the [`to_onnx`][lerax.export.onnx.to_onnx] function for converting
trained policies to the ONNX format for deployment in production runtimes.

Requires the ``onnx`` extra: ``pip install lerax[onnx]``
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import once
from jaxtyping import Array, Float

from ..policy.base_policy import AbstractPolicy

if TYPE_CHECKING:
    from collections.abc import Iterator

    import onnx as onnx_lib


@once.once
def register_is_finite_plugin() -> None:
    """
    Register the ``is_finite`` JAX primitive with jax2onnx.

    Maps ``jax.lax.is_finite_p`` to ``Not(Or(IsInf(x), IsNaN(x)))``.
    """
    from jax import core
    from jax2onnx.converter.typing_support import LoweringContextProtocol
    from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

    JaxprEqn = getattr(core, "JaxprEqn", Any)

    @register_primitive(
        jaxpr_primitive=jax.lax.is_finite_p.name,
        jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.is_finite.html",
        onnx=[
            {
                "component": "IsInf",
                "doc": "https://onnx.ai/onnx/operators/onnx__IsInf.html",
            },
            {
                "component": "IsNaN",
                "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
            },
            {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
            {
                "component": "Not",
                "doc": "https://onnx.ai/onnx/operators/onnx__Not.html",
            },
        ],
        since="v0.12.0",
        context="primitives.lax",
        component="is_finite",
        testcases=[],
    )
    class IsFinitePlugin(PrimitiveLeafPlugin):
        def lower(
            self,
            ctx: LoweringContextProtocol,
            eqn: JaxprEqn,
            *extra: Any,
            **kwargs: Any,
        ) -> None:
            x_var = eqn.invars[0]
            out_var = eqn.outvars[0]

            x_val = ctx.get_value_for_var(
                x_var, name_hint=ctx.fresh_name("isfinite_in")
            )

            is_inf = ctx.builder.IsInf(x_val, _outputs=[ctx.fresh_name("is_inf")])
            is_nan = ctx.builder.IsNaN(x_val, _outputs=[ctx.fresh_name("is_nan")])
            is_inf_or_nan = ctx.builder.Or(
                is_inf, is_nan, _outputs=[ctx.fresh_name("is_inf_or_nan")]
            )
            result = ctx.builder.Not(
                is_inf_or_nan, _outputs=[ctx.fresh_name("isfinite_out")]
            )

            out_spec = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("isfinite_out")
            )
            if getattr(out_spec, "type", None) is not None:
                result.type = out_spec.type
            if getattr(out_spec, "shape", None) is not None:
                result.shape = out_spec.shape
            ctx.bind_value_for_var(out_var, result)


@contextmanager
def suppress_error_if() -> Iterator[None]:
    """
    Temporarily replace ``eqx.error_if`` with a no-op during ONNX tracing.
    """
    original = eqx.error_if

    def _noop_error_if(x, *_args, **_kwargs):
        return x

    eqx.error_if = _noop_error_if
    try:
        yield
    finally:
        eqx.error_if = original


def to_onnx(
    policy: AbstractPolicy,
    *,
    output_path: str | Path | None = None,
    model_name: str = "lerax_policy",
    opset: int = 21,
) -> onnx_lib.ModelProto:
    """
    Export a policy's deterministic inference path to an ONNX model.

    Traces ``policy(None, observation)`` with ``key=None`` (deterministic mode)
    and converts the resulting JAX computation graph to ONNX format using
    ``jax2onnx``.

    The exported model maps a flat observation array to an action array.

    Args:
        policy: The trained policy to export.
        output_path: If provided, save the ONNX model to this file path.
        model_name: Name embedded in the ONNX model metadata.
        opset: Target ONNX opset version.

    Returns:
        The ONNX ModelProto object.

    Raises:
        ImportError: If jax2onnx is not installed.

    Example:
        ```python
        from lerax.export import to_onnx

        proto = to_onnx(policy, output_path="policy.onnx")
        ```
    """
    try:
        from jax2onnx import to_onnx as _jax2onnx
    except ImportError as exc:
        raise ImportError(
            "jax2onnx is required for ONNX export. "
            "Install it with: pip install lerax[onnx]"
        ) from exc

    register_is_finite_plugin()

    observation_shape = (policy.observation_space.flat_size,)

    def inference(observation: Float[Array, " observation_dim"]) -> Array:
        _, action = policy(None, observation)
        return action

    input_spec = jax.ShapeDtypeStruct(observation_shape, jnp.float32)

    with suppress_error_if():
        proto = _jax2onnx(
            inference,
            [input_spec],
            model_name=model_name,
            opset=opset,
            return_mode="proto",
        )

    if output_path is not None:
        import onnx

        onnx.save(proto, str(Path(output_path)))

    return proto
