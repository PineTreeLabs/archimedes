"""Shared basis-family builders for tests that sweep FunctionSpace behavior
across representative basis families.

Several test modules each construct a small set of ``FunctionSpace``
instances -- one per basis family ("modal"
:class:`~archimedes.approximation.OrthogonalPolynomialBasis`, "nodal"
:class:`~archimedes.approximation.LagrangeBasis`, "piecewise"
:class:`~archimedes.approximation.PiecewiseBasis`, "bspline"
:class:`~archimedes.approximation.BSplineBasis`, and occasionally "jacobi",
the same modal family under a different :class:`~archimedes.measure.Measure`)
-- and sweep a `space` fixture across whichever subset is relevant to what
they're testing.

Centralizing the construction here, rather than each file defining its own
builder dict, keeps the family set and how each family is built in one
place. The numeric knobs (`n_basis`, breakpoints, quadrature) still differ
per file: a product needs headroom for `n_1 + n_2 - 1` growth, a
vector-valued projection needs an explicit `quad_rule` so every family
represents the same low-degree components exactly, and so on. Callers pass
the knobs they need at the call site via :func:`family_space`/
:func:`family_builders`, instead of hand-rolling a new builder dict.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from archimedes.approximation import (
    FunctionSpace,
    LagrangeBasis,
    OrthogonalPolynomialBasis,
    PiecewiseBasis,
)
from archimedes.measure import JacobiMeasure, LegendreMeasure, Measure, UnitInterval
from archimedes.quadrature import Quadrature, gauss_lobatto


def lobatto_basis(n: int) -> LagrangeBasis:
    """Build a :class:`~archimedes.approximation.LagrangeBasis` on the
    ``n``-point Gauss-Lobatto reference nodes -- the element basis every
    "nodal" and "piecewise" builder below uses."""
    return LagrangeBasis(reference_nodes=gauss_lobatto(n).nodes)


def modal_space(
    n_basis: int,
    a: float = 0.0,
    b: float = 2.0,
    measure: Measure | None = None,
    quad_rule: Quadrature | None = None,
) -> FunctionSpace:
    """Build a global orthogonal-polynomial space (Legendre by default) on
    ``[a, b]``."""
    measure = LegendreMeasure() if measure is None else measure
    return FunctionSpace(
        OrthogonalPolynomialBasis(measure, n_basis),
        domain=UnitInterval.Parameters(a=a, b=b),
        reference_quad_rule=quad_rule,
    )


def nodal_space(
    n_basis: int,
    a: float = 0.0,
    b: float = 2.0,
    quad_rule: Quadrature | None = None,
) -> FunctionSpace:
    """Build a global Lagrange (Gauss-Lobatto node) space on ``[a, b]``."""
    return FunctionSpace(
        lobatto_basis(n_basis),
        domain=UnitInterval.Parameters(a=a, b=b),
        reference_quad_rule=quad_rule,
    )


def piecewise_space(
    element_n_basis: int,
    breakpoints: np.ndarray,
    a: float = 0.0,
    b: float = 2.0,
    continuity: int = 0,
    quad_rule: Quadrature | None = None,
) -> FunctionSpace:
    """Build a piecewise Lagrange space: ``element_n_basis``-node elements
    tiled across (reference-domain) ``breakpoints``."""
    return FunctionSpace(
        PiecewiseBasis(
            lobatto_basis(element_n_basis), breakpoints, continuity=continuity
        ),
        domain=UnitInterval.Parameters(a=a, b=b),
        reference_quad_rule=quad_rule,
    )


def bspline_space(
    degree: int,
    breakpoints: np.ndarray,
    quad_rule: Quadrature | None = None,
) -> FunctionSpace:
    """Build a clamped B-spline space on physical-domain ``breakpoints``."""
    return FunctionSpace.clamped_bspline(degree, breakpoints, quad_rule=quad_rule)


def family_builders(
    n_basis: int = 6,
    a: float = 0.0,
    b: float = 2.0,
    nodal_n_basis: int | None = None,
    breakpoints: np.ndarray | None = None,
    element_n_basis: int | None = None,
    continuity: int = 0,
    quad_rule: Quadrature | None = None,
    piecewise_quad_rule: Quadrature | None = None,
    bspline_degree: int = 3,
    bspline_breakpoints: np.ndarray | None = None,
    jacobi_measure: Measure | None = None,
) -> dict[str, Callable[[], FunctionSpace]]:
    """Build the ``{family_name: () -> FunctionSpace}`` registry, sized and
    placed by the keyword arguments given.

    Every caller picks the numeric knobs its own tests need -- this shares
    the *construction*, not one fixed set of numbers.

    Parameters
    ----------
    n_basis : int
        Size of the "modal"/"jacobi" global spaces, and the default for
        "nodal" if `nodal_n_basis` is not given.
    a, b : float
        Physical domain endpoints, shared by every family.
    nodal_n_basis : int, optional
        Size of the "nodal" space, if different from `n_basis` (e.g. a
        family sweep that needs a different degree per family to make its
        own operation exact). Defaults to `n_basis`.
    breakpoints : ndarray, optional
        Reference-domain breakpoints for "piecewise" (see
        :class:`~archimedes.approximation.PiecewiseBasis`). "piecewise" is
        only included in the returned registry if this is given.
    element_n_basis : int, optional
        Per-element size for "piecewise". Defaults to `nodal_n_basis`.
    continuity : int
        Continuity for "piecewise". Default 0 (:math:`C^0`).
    quad_rule : Quadrature, optional
        Explicit reference-domain quadrature shared by "modal", "jacobi",
        "nodal", and "bspline".
    piecewise_quad_rule : Quadrature, optional
        Explicit reference-domain quadrature for "piecewise" specifically,
        since a piecewise rule generally needs its own element structure
        (e.g. ``composite_quad``) rather than the plain rule above.
    bspline_degree : int
        Polynomial degree per B-spline element. Default 3 (cubic).
    bspline_breakpoints : ndarray, optional
        Physical-domain breakpoints for "bspline". Defaults to 4 points
        evenly spaced over ``[a, b]``.
    jacobi_measure : Measure, optional
        Measure for "jacobi". Default ``JacobiMeasure(1.5, 0.5)``.

    Returns
    -------
    dict[str, Callable[[], FunctionSpace]]
        One zero-argument builder per family name -- "modal", "jacobi",
        "nodal", "bspline", and (only if `breakpoints` is given) "piecewise".
    """
    nodal_n_basis = n_basis if nodal_n_basis is None else nodal_n_basis
    element_n_basis = nodal_n_basis if element_n_basis is None else element_n_basis
    jacobi_measure = (
        JacobiMeasure(1.5, 0.5) if jacobi_measure is None else jacobi_measure
    )
    bspline_breakpoints = (
        np.linspace(a, b, 4) if bspline_breakpoints is None else bspline_breakpoints
    )

    builders: dict[str, Callable[[], FunctionSpace]] = {
        "modal": lambda: modal_space(n_basis, a, b, quad_rule=quad_rule),
        "jacobi": lambda: modal_space(
            n_basis, a, b, measure=jacobi_measure, quad_rule=quad_rule
        ),
        "nodal": lambda: nodal_space(nodal_n_basis, a, b, quad_rule=quad_rule),
        "bspline": lambda: bspline_space(
            bspline_degree, bspline_breakpoints, quad_rule=quad_rule
        ),
    }
    if breakpoints is not None:
        builders["piecewise"] = lambda: piecewise_space(
            element_n_basis,
            breakpoints,
            a,
            b,
            continuity=continuity,
            quad_rule=piecewise_quad_rule,
        )
    return builders


def family_space(*names: str, **builder_kwargs):
    """Build a ``pytest.fixture`` parametrized over the given basis-family
    ``names``, each built from :func:`family_builders`.

    Usage, at module level::

        space = family_space("modal", "nodal", "piecewise", n_basis=6,
                              breakpoints=BREAKS)

    Parameters
    ----------
    *names : str
        Basis family names to sweep; must be a subset of the names
        :func:`family_builders` would return for `builder_kwargs`.
    **builder_kwargs
        Forwarded to :func:`family_builders`.
    """
    builders = family_builders(**builder_kwargs)
    unknown = sorted(set(names) - set(builders))
    if unknown:
        raise ValueError(
            f"unknown or unconfigured basis family names {unknown}; available "
            f"names for these builder_kwargs are {sorted(builders)}"
        )

    @pytest.fixture(params=sorted(names))
    def _space(request):
        return builders[request.param]()

    return _space
