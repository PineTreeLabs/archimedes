r"""B-spline basis on a general knot vector.

Evaluation follows de Boor's BSPLVB algorithm (*A Practical Guide to
Splines*, Revised Ed., Ch. X): the stable triangular recurrence that builds
all ``degree + 1`` nonzero B-splines at a point column by column, with no
special-casing for repeated knots and denominators bounded below by the
local knot spacing. Derivatives reuse the same recurrence at a lower order
followed by the classical knot-difference derivative formula (Ch. IX-X,
the basis-function analogue of BVALUE's stage-1 coefficient differencing),
applied one order at a time.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from archimedes._core._array_impl import SymbolicArray
from archimedes.measure import UnitInterval

from ._base import RIGHT, Basis, _check_side
from ._utils import _gather, _locate, _reference_breakpoints

__all__ = ["BSplineBasis"]


@dataclasses.dataclass(frozen=True)
class BSplineBasis(Basis):
    r"""B-splines :math:`\{N_{i,p}\}_{i=1}^n` on a general knot vector.

    ``knots`` is a nondecreasing sequence in _physical_ units. Unlike
    piecewise polynomials (e.g. Lagrange), a B-spline's knot vector has no
    canonical reference form to remap.

    The knot vector can contain any interior or end multiplicity from ``1``
    through ``degree + 1``, clamped or open. A knot of multiplicity
    ``degree + 1`` gives a true discontinuity; one of multiplicity
    ``m < degree + 1`` gives :math:`C^{degree - m}` continuity there.
    Periodic (closed-curve) B-splines are a different construction and are
    not supported.

    Outside the basic interval ``[knots[degree], knots[-1-degree]]``,
    :meth:`evaluate` extrapolates via the boundary span's polynomial piece.

    Parameters
    ----------
    degree : int
        Polynomial degree of each piece. ``0`` gives piecewise-constant
        indicator functions.
    knots : array_like
        Nondecreasing knot vector, physical units, length
        ``n_basis + degree + 1``.

    Raises
    ------
    ValueError
        If ``degree < 0``; if ``knots`` is not 1-D, not nondecreasing, or
        has fewer than ``2 * degree + 2`` entries; if any value repeats
        more than ``degree + 1`` times; or if the implied basic interval
        ``[knots[degree], knots[-1-degree]]`` is degenerate.

    See Also
    --------
    FunctionSpace.bspline : Convenience constructor deriving ``Parameters``
        automatically from an explicit knot vector.
    FunctionSpace.clamped_bspline : Convenience constructor building a
        clamped knot vector from physical breakpoints (the common case).
    """

    degree: int
    knots: np.ndarray

    def __post_init__(self):
        if self.degree < 0:
            raise ValueError(f"degree must be >= 0, got {self.degree}")

        knots = np.asarray(self.knots, dtype=float)
        if knots.ndim != 1:
            raise ValueError(f"knots must be 1-D, got shape {knots.shape}")
        if len(knots) < 2 * self.degree + 2:
            raise ValueError(
                f"knots must have at least 2*degree+2 = {2 * self.degree + 2} "
                f"entries for degree={self.degree}, got {len(knots)}"
            )
        if np.any(np.diff(knots) < 0):
            raise ValueError("knots must be nondecreasing")

        _, counts = np.unique(knots, return_counts=True)
        if counts.max() > self.degree + 1:
            raise ValueError(
                f"a knot may repeat at most degree + 1 = {self.degree + 1} "
                f"times, got a value repeated {int(counts.max())} times"
            )
        if knots[self.degree] >= knots[len(knots) - 1 - self.degree]:
            raise ValueError(
                f"knots imply a degenerate basic interval "
                f"[{knots[self.degree]}, {knots[len(knots) - 1 - self.degree]}]"
            )

        object.__setattr__(self, "knots", knots)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BSplineBasis):
            return NotImplemented
        return self.degree == other.degree and np.array_equal(self.knots, other.knots)

    def __hash__(self) -> int:
        return hash((type(self), self.degree, self.knots.tobytes()))

    # --- structure ---

    @property
    def n_basis(self) -> int:
        return len(self.knots) - self.degree - 1

    @property
    def Parameters(self) -> type:  # noqa: N802
        return UnitInterval.Parameters

    @property
    def required_breakpoints(self) -> np.ndarray:
        """Distinct interior knots, on the reference domain ``[-1, 1]``"""
        interior = self.knots[self.degree : len(self.knots) - self.degree]
        distinct = np.unique(interior)
        _, _, ref = _reference_breakpoints(distinct)
        return ref

    def default_quadrature(self):
        """``degree + 1`` Gauss-Legendre points per distinct knot span."""
        from archimedes.quadrature import composite_quad, gauss_legendre

        rule = gauss_legendre(self.degree + 1)
        return composite_quad(rule, self.required_breakpoints)

    def boundary_dofs(self, order: int = 0) -> tuple[int | None, int | None]:
        """Indices of the degrees of freedom that *are* the ``order``-th
        derivative at the left and right ends of the domain.

        Returns ``(0, n_basis - 1)`` when ``knots`` is clamped (multiplicity
        ``degree + 1`` at both ends, so the first/last coefficients *are*
        the endpoint values); ``(None, None)`` otherwise, since an interior
        B-spline coefficient is a control point, not a value the basis
        interpolates at any point of its own support.
        """
        if order != 0:
            return (None, None)
        p = self.degree
        left_clamped = bool(np.all(self.knots[: p + 1] == self.knots[0]))
        right_tail = self.knots[len(self.knots) - p - 1 :]
        right_clamped = bool(np.all(right_tail == self.knots[-1]))
        if left_clamped and right_clamped:
            return (0, self.n_basis - 1)
        return (None, None)

    def _derivative_basis(self, deriv: int = 1) -> "BSplineBasis":
        """Derivative of the B-spline basis.

        Uses the fact that differentiating a B-spline lowers its degree
        by one and removes one knot from each end.
        """
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")
        if deriv == 0:
            return self
        if deriv > self.degree:
            raise ValueError(
                f"deriv={deriv} is at or past the degree of a degree-"
                f"{self.degree} B-spline basis; the derivative is "
                f"identically zero and has no space of its own. Use "
                f"`f(x, deriv={deriv})` if the zero values are what you want."
            )
        return BSplineBasis(
            self.degree - deriv, self.knots[deriv : len(self.knots) - deriv]
        )

    def _integral_basis(self, order: int = 1) -> "BSplineBasis":
        """Anti-derivative of the B-spline basis

        Degree ``degree + order`` on the knot vector with each end knot
        duplicated ``order`` more times. Interior knots (and so interior
        continuity) are untouched, since integrating raises smoothness by
        exactly one order everywhere; only the two ends need an extra
        degree of freedom each to hold the new degree.
        """
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if order == 0:
            return self
        knots = self.knots
        for _ in range(order):
            knots = np.concatenate([knots[:1], knots, knots[-1:]])
        return BSplineBasis(self.degree + order, knots)

    # --- evaluation ---

    def _local_values(self, x, deriv: int, side: str):
        """The ``degree + 1`` nonzero (``deriv``-th derivative) B-spline
        values at each point of ``x``, as a list of ``degree + 1`` arrays
        shape ``(npts,)``, together with ``base`` (shape ``(npts,)``): the
        global index of the first (list index 0) local function.

        Caller must ensure ``0 <= deriv <= degree``. Implements BSPLVB
        (de Boor Ch. X) for ``deriv=0``; for ``deriv > 0``, first runs
        BSPLVB to depth ``degree - deriv`` (giving the local values of the
        order-``(degree+1-deriv)`` B-splines), then applies the classical
        single-order derivative identity

        .. math::
            \\frac{d}{dx} N_{i,q+1}(x) = q \\left(
                \\frac{N_{i,q}(x)}{t_{i+q}-t_i} -
                \\frac{N_{i+1,q}(x)}{t_{i+q+1}-t_{i+1}}
            \\right)

        ``deriv`` times in a row -- valid at every order since the identity
        is linear in :math:`N`, so differentiating it again reproduces the
        same form one order higher (the design-matrix analogue of BVALUE's
        stage-1 coefficient differencing, Ch. X).
        """
        knots = self.knots
        p = self.degree
        symbolic = isinstance(x, SymbolicArray)
        npts = np.shape(x)[0]

        # `_locate`'s own `side="left"` steps back exactly one *array*
        # index at a knot, which is correct for `PiecewiseBasis` (breakpoints
        # always strictly increasing) but not here: a B-spline knot can
        # repeat up to `degree + 1` times, so landing exactly on a knot may
        # require stepping back over an entire *run* of repeated values, not
        # just one. Always locate from the right first (always a
        # nondecreasing-knots-safe, positive-width span by construction),
        # then, for the left limit, keep stepping back while sitting exactly
        # on `x` -- bounded by the maximum allowed multiplicity, so this is
        # a fixed number of branch-free steps, not an unbounded search.
        mu = _locate(knots, x, symbolic, RIGHT)
        if side != RIGHT:
            for _ in range(p + 1):
                at_x = _gather(knots, mu, symbolic, npts) == x
                mu = np.maximum(mu - at_x, 0)
        mu = np.minimum(np.maximum(mu, p), self.n_basis - 1)

        depth = p - deriv
        biatx = [np.ones_like(x)]
        if depth > 0:
            # `delta_r`/`delta_l` are this basis's `\delta^R`/`\delta^L`
            # (de Boor's BSPLVB notation): `delta_r[i-1] = t_{mu+i} - x`,
            # `delta_l[i-1] = x - t_{mu+1-i}`, for i=1..depth. Both depend
            # only on `mu`/`x`, not on the outer `j` loop, so each is built
            # once and reused as `j` grows -- the same reuse BSPLVB itself
            # relies on.
            delta_r = [
                _gather(knots, mu + r, symbolic, npts) - x for r in range(1, depth + 1)
            ]
            delta_l = [
                x - _gather(knots, mu + 1 - r, symbolic, npts)
                for r in range(1, depth + 1)
            ]
            for j in range(1, depth + 1):
                saved = np.zeros_like(x)
                new_biatx = [None] * (j + 1)
                for i in range(1, j + 1):
                    denom = delta_r[i - 1] + delta_l[j - i]
                    term = biatx[i - 1] / denom
                    new_biatx[i - 1] = saved + delta_r[i - 1] * term
                    saved = delta_l[j - i] * term
                new_biatx[j] = saved
                biatx = new_biatx

        base = mu - depth  # global index of biatx[0]
        for _ in range(deriv):
            q = len(biatx)
            new_biatx = [None] * (q + 1)
            for j in range(q + 1):
                if j >= 1:
                    lo = _gather(knots, base - 1 + j, symbolic, npts)
                    hi = _gather(knots, base - 1 + j + q, symbolic, npts)
                    left = biatx[j - 1] / (hi - lo)
                else:
                    left = np.zeros_like(x)
                if j <= q - 1:
                    lo = _gather(knots, base + j, symbolic, npts)
                    hi = _gather(knots, base + j + q, symbolic, npts)
                    right = biatx[j] / (hi - lo)
                else:
                    right = np.zeros_like(x)
                new_biatx[j] = q * (left - right)
            biatx = new_biatx
            base = base - 1

        return biatx, base

    def evaluate(self, x, deriv: int = 0, a=None, b=None, side: str = RIGHT):
        """Evaluate all ``n_basis`` B-splines at ``x``.

        ``a``/``b`` are accepted (every ``Basis.evaluate`` must accept the
        domain kwargs ``FunctionSpace`` forwards) but ignored -- ``knots``
        are already physical, so there is nothing to remap; see the class
        docstring.
        """
        _check_side(side)
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")

        n_basis = self.n_basis
        if deriv > self.degree:
            return np.stack([np.zeros_like(x)] * n_basis, axis=-1)

        biatx, base = self._local_values(x, deriv, side)
        col = np.arange(n_basis)

        total = None
        for k, values in enumerate(biatx):
            global_idx = base + k
            mask = global_idx[:, None] == col[None, :]
            contribution = np.where(mask, values[:, None], 0.0)
            total = contribution if total is None else total + contribution
        return total

    def evaluate_expansion(
        self, coefficients, x, deriv: int = 0, side: str = RIGHT, **domain_kwargs
    ):
        """Evaluate :math:`\\sum_i c_i \\, \\N_i(x)` (or its derivative) directly."""
        # Gathers only the ``degree + 1`` locally relevant coefficients per point
        # rather than materializing the full ``(npts, n_basis)`` design matrix: the
        # local-support analogue of PiecewiseBasis.evaluate_expansion.
        _check_side(side)
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")

        symbolic = isinstance(x, SymbolicArray) or isinstance(
            coefficients, SymbolicArray
        )
        npts = np.shape(x)[0]
        vector_valued = np.ndim(coefficients) > 1

        if deriv > self.degree:
            if vector_valued:
                m = np.shape(coefficients)[1]
                return np.stack([np.zeros_like(x)] * m, axis=-1)
            return np.zeros_like(x)

        biatx, base = self._local_values(x, deriv, side)

        total = None
        for k, values in enumerate(biatx):
            c_k = _gather(coefficients, base + k, symbolic, npts)
            term = values[:, None] * c_k if vector_valued else values * c_k
            total = term if total is None else total + term
        return total
