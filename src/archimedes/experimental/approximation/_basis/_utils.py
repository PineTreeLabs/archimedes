"""Shared symbolic-safe indexing helpers for knot/breakpoint-based families.

Utilities for locating which interval of a nondecreasing array owns a point
and gathering rows by a (possibly symbolic) integer index.
"""

from __future__ import annotations

import casadi as cs
import numpy as np

from archimedes._core._array_impl import SymbolicArray, _unwrap_sym_array

from ._base import RIGHT, _check_side

__all__ = ["_as_mx", "_gather", "_locate"]


def _as_mx(value):
    """Unwrap to a CasADi MX. ``cs.low`` and symbolic-index gathers are
    MX-only, and a constant must be promoted from DM before it can be
    indexed by a symbolic expression."""
    if isinstance(value, SymbolicArray):
        return _unwrap_sym_array(value)
    return cs.MX(cs.DM(np.asarray(value, dtype=float)))


def _locate(knots, x, symbolic: bool, side: str = RIGHT):
    """Index of the interval owning each point of ``x``, by coordinate.

    With ``side="right"`` (the default) ownership is half-open ``[lo, hi)``,
    so a point on a breakpoint belongs to the interval above it -- the limit
    from the right. ``side="left"`` gives ``(lo, hi]`` and the limit from
    the left. Both clamp out-of-range points into the end intervals.

    ``cs.low`` is CasADi's ``std::lower_bound`` and has exactly the
    right-sided semantics; ``searchsorted(..., "right") - 1`` is its NumPy
    equivalent. The left-sided variant steps back one interval at points that
    land exactly on a knot, which costs one comparison against a value the
    caller is gathering anyway.

    This is the coordinate-only path, used for user-supplied points. Where
    provenance exists (quadrature nodes) ``PiecewiseBasis._evaluate_at_nodes``
    uses it instead and no convention is needed.

    A caller whose valid range is a strict subset of ``[0, len(knots) - 2]``
    (e.g. a B-spline's basic interval, which excludes the padding knots
    outside ``[knots[degree], knots[-1-degree]]``) should clip the result
    itself; this always returns an index into the *full* ``knots`` array.
    """
    _check_side(side)
    n_elements = len(knots) - 1
    if symbolic:
        index = SymbolicArray(
            cs.low(_as_mx(knots), _as_mx(x)), shape=np.shape(x), dtype=int
        )
    else:
        index = np.clip(
            np.searchsorted(np.asarray(knots), np.asarray(x), side="right") - 1,
            0,
            n_elements - 1,
        )
    if side == RIGHT:
        return index
    # On a knot, back up one interval; `np.where` keeps this branch-free so it
    # traces, and the max() guards the first interval's left end.
    on_knot = _gather(knots, index, symbolic, np.shape(x)[0]) == x
    return np.maximum(index - on_knot, 0)


def _gather(values, index, symbolic: bool, npts: int):
    """``values[index]`` (rows, if ``values`` is 2-D) for a possibly
    symbolic integer ``index``."""
    if not symbolic:
        return values[index]
    shape = (npts,) if np.ndim(values) == 1 else (npts, np.shape(values)[1])
    gathered = _as_mx(values)[_as_mx(index), :]
    return SymbolicArray(gathered, shape=shape, dtype=float)
