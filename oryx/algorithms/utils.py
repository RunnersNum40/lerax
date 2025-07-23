import inspect
from functools import partial, wraps
from typing import Any, Callable, Iterable

import jax
import numpy as np
from jax import numpy as jnp


def debug_wrapper[**InType](
    func: Callable[InType, Any], ordered: bool = False
) -> Callable[InType, None]:
    """Return a JIT‑safe version of *func*."""

    @wraps(func)
    def wrapped(*args: InType.args, **kwargs: InType.kwargs) -> None:
        jax.debug.callback(func, *args, **kwargs, ordered=ordered)

    return wrapped


def debug_with_numpy_wrapper[**InType](
    func: Callable[InType, Any], ordered: bool = False
) -> Callable[InType, None]:
    """
    Like ``debug_wrapper`` but converts every ``jax.Array``/``jnp.ndarray`` argument
    to a plain ``numpy.ndarray`` before calling *func*.
    """

    @partial(debug_wrapper, ordered=ordered)
    @wraps(func)
    def wrapped(*args: InType.args, **kwargs: InType.kwargs) -> None:
        args, kwargs = jax.tree_util.tree_map(
            lambda x: np.asarray(x) if isinstance(x, (jax.Array, jnp.ndarray)) else x,
            (args, kwargs),
        )
        func(*args, **kwargs)

    return wrapped


def patch_methods[**P, T](
    target: T | type[T],
    *,
    decorator: Callable[
        [Callable[P, Any], bool], Callable[P, Any]
    ] = debug_with_numpy_wrapper,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    ordered: bool = False,
) -> T | type[T]:
    """
    Replace every selected method of *target* (class or instance) with
    ``decorator(original, ordered=ordered)`` and return *target*.
    """
    if include is not None:
        names = include
    else:
        pred = lambda n, m: callable(m) and not n.startswith("_")
        members = inspect.getmembers(target, predicate=None)
        names = (n for n, m in members if pred(n, m))

    for name in names:
        if exclude and name in exclude:
            continue
        try:
            orig = getattr(target, name)
        except AttributeError:
            continue
        wrapped = decorator(orig, ordered)
        # class patch or instance
        if inspect.isclass(target):
            setattr(target, name, wrapped)
        else:
            setattr(target, name, wrapped.__get__(target, target.__class__))
    return target
