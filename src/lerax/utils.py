from __future__ import annotations

import hashlib
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jaxtyping import ArrayLike, Bool


def filter_cond[**ParamType, RetType](
    pred: Bool[ArrayLike, ""],
    true_fun: Callable[ParamType, RetType],
    false_fun: Callable[ParamType, RetType],
    *args: ParamType.args,
    **kwargs: ParamType.kwargs,
) -> RetType:
    """
    Like `lax.cond` but handles non-array leaves (e.g. activation functions
    inside Equinox modules).

    Note:
        The non-array leaves of the outputs of `true_fun` and `false_fun`
        must be identical.

    Args:
        pred: A boolean scalar determining which branch to select.
        true_fun: A callable to be executed if `pred` is True.
        false_fun: A callable to be executed if `pred` is False.
        args: Positional arguments to be passed to both `true_fun` and
            `false_fun`.
        kwargs: Keyword arguments to be passed to both `true_fun` and
            `false_fun`.

    Returns:
        The result of `true_fun` if `pred` is True, else `false_fun`,
        with array leaves selected via `lax.cond`.

    Raises:
        ValueError: If the non-array leaves of the outputs of `true_fun` and
            `false_fun` are not identical.
    """
    true_result = true_fun(*args, **kwargs)
    false_result = false_fun(*args, **kwargs)

    result_arrays, result_static = eqx.partition(
        (true_result, false_result), eqx.is_array
    )

    if not eqx.tree_equal(result_static[0], result_static[1]):
        raise ValueError(
            "Non-array leaves of true_fun and false_fun outputs must be identical."
            f"Got\n{result_static[0]}\nand\n{result_static[1]}"
        )

    return_result = lax.cond(pred, lambda: result_arrays[0], lambda: result_arrays[1])
    return eqx.combine(return_result, result_static[0])


def filter_scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, Y]:
    """
    An easier to use version of `lax.scan`. All JAX and Numpy arrays are
    traced, and only non-array parts of the carry are static.

    Args:
        f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
            that ``f`` accepts two arguments where the first is a value of the loop
            carry and the second is a slice of ``xs`` along its leading axis, and that
            ``f`` returns a pair where the first element represents a new value for
            the loop carry and the second represents a slice of the output.
        init: an initial loop carry value of type ``c``, which can be a scalar,
            array, or any pytree (nested Python tuple/list/dict) thereof, representing
            the initial loop carry value. This value must have the same structure as
            the first element of the pair returned by ``f``.
        xs: the value of type ``[a]`` over which to scan along the leading axis,
            where ``[a]`` can be an array or any pytree (nested Python
            tuple/list/dict) thereof with consistent leading axis sizes.
        length: optional integer specifying the number of loop iterations, which
            must agree with the sizes of leading axes of the arrays in ``xs`` (but can
            be used to perform scans where no input ``xs`` are needed).
        reverse: optional boolean specifying whether to run the scan iteration
            forward (the default) or in reverse, equivalent to reversing the leading
            axes of the arrays in both ``xs`` and in ``ys``.
        unroll: optional non-negative int or bool specifying, in the underlying
            operation of the scan primitive, how many scan iterations to unroll within
            a single iteration of a loop. If an integer is provided, it determines how
            many unrolled loop iterations to run within a single rolled iteration of
            the loop. `unroll=0` unrolls the entire loop.
            If a boolean is provided, it will determine if the loop is
            completely unrolled (i.e. `unroll=True`) or left completely rolled (i.e.
            `unroll=False`).
        _split_transpose: experimental optional bool specifying whether to further
            split the transpose into a scan (computing activation gradients), and a
            map (computing gradients corresponding to the array arguments). Enabling
            this may increase memory requirements, and so is an experimental feature
            that may evolve or even be rolled back.

    Returns:
        A pair of type ``(c, [b])`` where the first element represents the final
        loop carry value and the second element represents the stacked outputs of
        the second output of ``f`` when scanned over the leading axis of the inputs.
    """
    init_arr, static = eqx.partition(init, eqx.is_array)

    def _f(carry_arr, x):
        carry = eqx.combine(carry_arr, static)
        carry, y = f(carry, x)
        new_carry_arr, new_static = eqx.partition(carry, eqx.is_array)
        assert eqx.tree_equal(static, new_static), (
            "Non-array carry of filter_scan must not change."
        )
        return new_carry_arr, y

    carry_arr, ys = lax.scan(
        f=_f,
        init=init_arr,
        xs=xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
    )
    return eqx.combine(carry_arr, static), ys


def callback_wrapper[**InType](
    func: Callable[InType, Any], ordered: bool = False, partitioned: bool = False
) -> Callable[InType, None]:
    """
    Return a JIT‑safe version of *func*.

    Wraps *func* in a `jax.debug.callback` so that it can be used inside JIT‑compiled
    code.

    Args:
        func: The callback function to wrap.
        ordered: Whether to enforce ordered execution of callbacks.
        partitioned: If True, then print local shards only; this option avoids an
            all-gather of the operands. If False, print with logical operands; this
            option requires an all-gather of operands first.

    Returns:
        A wrapped version of *func* that is JIT-safe.
    """

    def _callback(*args: InType.args, **kwargs: InType.kwargs) -> None:
        func(*args, **kwargs)

    @wraps(func)
    def wrapped(*args: InType.args, **kwargs: InType.kwargs) -> None:
        jax.debug.callback(
            _callback, *args, ordered=ordered, partitioned=partitioned, **kwargs
        )

    return wrapped


def callback_with_numpy_wrapper(
    func: Callable[..., Any], ordered: bool = False, partitioned: bool = False
) -> Callable[..., None]:
    """
    Like `debug_wrapper` but converts every jax.Array/`jnp.ndarray` argument
    to a plain `numpy.ndarray` before calling *func*.

    It is impossible with Python's current type system to express the
    transformation so parameter information is lost.

    Args:
        func: The callback function to wrap.
        ordered: Whether to enforce ordered execution of callbacks.
        partitioned: If True, then print local shards only; this option avoids an
            all-gather of the operands. If False, print with logical operands; this
            option requires an all-gather of operands first.

    Returns:
        A wrapped version of *func* that converts array arguments to numpy
        arrays and is JIT-safe.
    """

    @partial(callback_wrapper, ordered=ordered, partitioned=partitioned)
    @wraps(func)
    def wrapped(*args, **kwargs) -> None:
        args, kwargs = jax.tree.map(
            lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, (args, kwargs)
        )
        func(*args, **kwargs)

    return wrapped


def callback_with_list_wrapper(
    func: Callable[..., Any], ordered: bool = False, partitioned: bool = False
) -> Callable[..., None]:
    """
    Like `debug_wrapper` but converts every jax.Array/`jnp.ndarray` argument
    to a plain list before calling *func*.

    It is impossible with Python's current type system to express the
    transformation so parameter information is lost.

    Args:
        func: The callback function to wrap.
        ordered: Whether to enforce ordered execution of callbacks.
        partitioned: If True, then print local shards only; this option avoids an
            all-gather of the operands. If False, print with logical operands; this
            option requires an all-gather of operands first.

    Returns:
        A wrapped version of *func* that converts array arguments to lists and
        is JIT-safe.
    """

    @partial(callback_wrapper, ordered=ordered, partitioned=partitioned)
    @wraps(func)
    def wrapped(*args, **kwargs) -> None:
        args, kwargs = jax.tree.map(
            lambda x: (
                np.asarray(x).tolist()
                if isinstance(x, (jnp.ndarray, np.ndarray))
                else x
            ),
            (args, kwargs),
        )
        func(*args, **kwargs)

    return wrapped


print_callback = callback_with_list_wrapper(print, ordered=True)


def unstack_pytree[T](tree: T, *, axis: int = 0) -> Sequence[T]:
    """
    Split a stacked pytree along `axis` into a tuple of pytrees with the same
    structure.

    Args:
        tree: A pytree with array leaves stacked along `axis`.
        axis: The axis along which to unstack the arrays.

    Returns:
        A sequence of pytrees with the same structure, each corresponding to one
        slice along `axis`.
    """

    times = jnp.array(jax.tree.leaves(jax.tree.map(lambda x: x.shape[axis], tree)))
    tree = eqx.error_if(
        tree,
        ~times == times[0],
        "All leaves must have the same size along the specified axis.",
    )

    outer_structure = jax.tree.structure(tree)
    unstacked = jax.tree.map(partial(jnp.unstack, axis=axis), tree)
    transposed = jax.tree.transpose(outer_structure, None, unstacked)
    return transposed


def polyak_average[T: eqx.Module](online: T, target: T, tau: float) -> T:
    """
    Polyak-average the parameters of two modules.

    Returns a new module whose inexact-array leaves are
    ``tau * online + (1 - tau) * target``, with all other leaves
    taken from ``online``.

    Args:
        online: The online (source) module.
        target: The target module to update towards.
        tau: Interpolation coefficient in ``[0, 1]``.

    Returns:
        The updated target module.
    """
    online_arrays, online_static = eqx.partition(online, eqx.is_inexact_array)
    target_arrays, _ = eqx.partition(target, eqx.is_inexact_array)
    updated = jax.tree.map(
        lambda o, t: tau * o + (1 - tau) * t,
        online_arrays,
        target_arrays,
    )
    return eqx.combine(updated, online_static)


def _structure_hash(tree: Any) -> bytes:
    """
    Return a 32-byte digest of the static (non-array) structure of *tree*.

    Two PyTrees produce the same digest iff they have the same `jax.tree`
    structure and the same static leaves (class identity, hyperparameters,
    activation functions, etc.). Array values do not affect the digest.
    """
    return hashlib.sha256(repr(jax.tree.structure(tree)).encode()).digest()


class Serializable(eqx.Module):
    def serialize(self, path: str | Path) -> None:
        """
        Serialize the model to the specified path.

        Writes a 32-byte structural fingerprint followed by the Equinox
        leaf data, so `deserialize` can verify that the skeleton it builds
        matches what was saved.

        Args:
            path: The path to serialize to. The ``.eqx`` suffix is appended
                if missing.
        """
        path = Path(path)
        if path.suffix != ".eqx":
            path = path.with_suffix(".eqx")
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(_structure_hash(self))
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def deserialize[**Params, ClassType](
        cls: Callable[Params, ClassType],
        path: str | Path,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ClassType:
        """
        Deserialize the model from the specified path.

        The constructor arguments must reproduce the same static structure
        (class, hyperparameters, network shapes, activations, ...) that the
        model had when it was serialized. A 32-byte fingerprint stored in
        the file is verified before loading; mismatches raise `ValueError`
        instead of silently loading arrays into the wrong skeleton.

        Args:
            path: The path to deserialize from.
            *args: Additional arguments to pass to the class constructor.
            **kwargs: Additional keyword arguments to pass to the class
                constructor.

        Returns:
            The deserialized model.

        Raises:
            ValueError: If the structural fingerprint of the rebuilt
                skeleton does not match the one stored in the file.
        """
        path = Path(path)
        if path.suffix != ".eqx":
            path = path.with_suffix(".eqx")
        skeleton = eqx.filter_eval_shape(cls, *args, **kwargs)
        expected = _structure_hash(skeleton)
        with open(path, "rb") as f:
            stored = f.read(32)
            if stored != expected:
                raise ValueError(
                    f"Structural fingerprint mismatch loading {path}: the "
                    "constructor arguments produce a different skeleton than "
                    "the one that was serialized."
                )
            return eqx.tree_deserialise_leaves(f, skeleton)
