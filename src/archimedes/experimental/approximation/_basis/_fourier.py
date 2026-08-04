"""Fourier (trigonometric) basis on a periodic interval."""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from archimedes.measure import LegendreMeasure, UnitInterval

from ._basis import RIGHT, Basis, _check_side

__all__ = ["FourierBasis"]

_PRODUCT_KIND: dict[tuple[str, str], Literal["cosine", "sine"]] = {
    ("cosine", "cosine"): "cosine",
    ("sine", "sine"): "cosine",
    ("cosine", "sine"): "sine",
    ("sine", "cosine"): "sine",
}


def _interleave_matrix(n: int) -> np.ndarray:
    """Static permutation matrix mapping ``[cos_1..cos_n, sin_1..sin_n]``
    (columns concatenated block-wise) to ``[cos_1, sin_1, ..., cos_n,
    sin_n]`` (interleaved) under right-multiplication: ``combined @
    _interleave_matrix(n)``. Pure NumPy -- built once per call from ``n``
    alone, never from traced data."""
    perm = np.empty(2 * n, dtype=int)
    perm[0::2] = np.arange(n)
    perm[1::2] = np.arange(n, 2 * n)
    return np.eye(2 * n)[:, perm]


@dataclasses.dataclass(frozen=True)
class FourierBasis(Basis):
    r"""Trigonometric basis on a periodic interval.

    :math:`\{1, \cos(\theta), \sin(\theta), \ldots,
    \cos(N\theta), \sin(N\theta)\}` on :math:`[a, b]`, where :math:`a` and :math:`b`
    are identified as the same point.

    **``kind``** selects which trigonometric family:

    - ``"full"`` (default): :math:`\{1, \cos(\theta), \sin(\theta),
      \ldots, \cos(N\theta), \sin(N\theta)\}`, so ``n_basis = 2N + 1``
      (must be odd).
    - ``"cosine"``: :math:`\{1, \cos(\theta), \ldots, \cos(N\theta)\}`,
      ``n_basis = N + 1``.
    - ``"sine"``: :math:`\{\sin(\theta), \ldots, \sin(N\theta)\}` (no
      constant term -- :math:`\sin(0) = 0` is trivial), ``n_basis = N``.

    :attr:`max_mode` gives :math:`N` uniformly across all three.

    **Normalization** mirrors :attr:`Basis.density` exactly as
    :class:`OrthogonalPolynomialBasis` does, reusing
    ``LegendreMeasure().mass(a, b)`` (:math:`= \mathrm{scale} \cdot 2 = b -
    a`, the period length :math:`L`) as the raw-weight mass: with
    ``mass = 1.0 if density else LegendreMeasure().mass(a, b)``, the
    constant term is :math:`1/\sqrt{\mathrm{mass}}` and each :math:`\cos`/
    :math:`\sin` term is :math:`\sqrt{2/\mathrm{mass}} \, \cos(k\theta)` /
    :math:`\sqrt{2/\mathrm{mass}} \, \sin(k\theta)`. ``density=False`` (the
    default) gives a basis orthonormal under the raw Lebesgue weight on
    :math:`[a, b]`; ``density=True`` gives one orthonormal under the
    *probability* density :math:`1/L`.

    **Derivatives** are closed-form and exact to arbitrary order via the
    cyclic identity :math:`d^m \cos(k\theta)/dx^m = (k\omega)^m
    \cos(k\theta + m\pi/2)` (and the same for :math:`\sin`) -- there is no
    recurrence or differentiation matrix.

    **Integrals.** ``"full"``/``"cosine"`` both contain the constant basis
    function, whose antiderivative is a non-periodic linear ramp with no
    representation in any Fourier-type space, so its antiderivative cannot
    be represented in the basis. ``"sine"`` has no constant term, so its
    first integral is well-defined, but higher orders raise the same error.

    Parameters
    ----------
    n_basis : int
        Number of basis functions; see the ``kind``-dependent conventions
        above. Must be ``>= 1``; ``kind="full"`` additionally requires an
        odd value.
    kind : {"full", "cosine", "sine"}, optional
        Which trigonometric family. Default ``"full"``.
    density : bool, optional
        Normalize against the probability density :math:`1/L` instead of
        the raw weight; see :attr:`Basis.density`. Default ``False``.
    """

    n_basis: int
    kind: Literal["full", "cosine", "sine"] = "full"
    density: bool = False

    def __post_init__(self):
        if self.kind not in ("full", "cosine", "sine"):
            raise ValueError(
                f"kind must be 'full', 'cosine', or 'sine', got {self.kind!r}"
            )
        if self.n_basis < 1:
            raise ValueError(f"n_basis must be >= 1, got {self.n_basis}")
        if self.kind == "full" and self.n_basis % 2 == 0:
            raise ValueError(
                f"kind='full' requires an odd n_basis (a constant term plus "
                f"paired cos/sin terms, n_basis = 2*max_mode + 1), got "
                f"n_basis={self.n_basis}"
            )

    @property
    def max_mode(self) -> int:
        """Highest mode number :math:`N` present in this basis."""
        if self.kind == "full":
            return (self.n_basis - 1) // 2
        if self.kind == "cosine":
            return self.n_basis - 1
        return self.n_basis  # "sine"

    @property
    def measures(self) -> tuple[LegendreMeasure]:
        """Orthogonal under the uniform (Lebesgue) weight."""
        return (LegendreMeasure(),)

    @property
    def Parameters(self) -> type:  # noqa: N802
        return UnitInterval.Parameters

    def default_quadrature(self):
        """Periodic-trapezoidal rule of ``2 * max_mode + 1`` points.

        Exact for trigonometric polynomials of mode :math:`\\leq 2 \\cdot
        \\mathrm{max\\_mode}`, which covers this basis's own mass-matrix
        integrand (products of modes up to :math:`\\mathrm{max\\_mode}`
        each).
        """
        from archimedes.quadrature import periodic_trapezoidal

        return periodic_trapezoidal(2 * self.max_mode + 1)

    def _product_basis(self, other):
        """Product-to-sum closure table (``N = self.max_mode +
        other.max_mode``):

        - either operand ``"full"`` -> ``"full"``, mode ``N``
        - ``"cosine" x "cosine"`` -> ``"cosine"``, mode ``N``
        - ``"sine" x "sine"`` -> ``"cosine"``, mode ``N`` (not ``"sine"``:
          :math:`\\sin(a)\\sin(b) = \\tfrac12[\\cos(a-b) - \\cos(a+b)]` has
          no sine term, and includes a DC term whenever the two modes
          match)
        - ``"cosine" x "sine"`` (either order) -> ``"sine"``, mode ``N``

        Requires the same ``density``, matching
        :meth:`OrthogonalPolynomialBasis._product_basis`.
        """
        if not isinstance(other, FourierBasis):
            raise ValueError(
                f"cannot form a product basis between "
                f"{type(self).__name__} and {type(other).__name__}"
            )
        if self.density != other.density:
            raise ValueError(
                f"product requires the same normalization, got "
                f"density={self.density} and density={other.density}"
            )
        N = self.max_mode + other.max_mode
        if self.kind == "full" or other.kind == "full":
            return FourierBasis(2 * N + 1, kind="full", density=self.density)
        kind = _PRODUCT_KIND[(self.kind, other.kind)]
        n_basis = N + 1 if kind == "cosine" else N
        return FourierBasis(n_basis, kind=kind, density=self.density)

    def _derivative_basis(self, deriv=1):
        """Cycles by ``deriv mod 2``.
        """
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")
        if self.kind == "full" or deriv % 2 == 0:
            return self
        if self.kind == "cosine":
            if self.max_mode == 0:
                raise ValueError(
                    "kind='cosine' with n_basis=1 is just the constant "
                    "function; its derivative is identically zero and has "
                    "no (odd-order) space of its own. Use `f(x, "
                    "deriv=...)` if the zero values are what you want."
                )
            return FourierBasis(self.max_mode, kind="sine", density=self.density)
        return FourierBasis(self.max_mode + 1, kind="cosine", density=self.density)

    def _integral_basis(self, order=1):
        """``"full"``/``"cosine"`` raise -- both contain the constant/DC
        basis function, whose antiderivative is a non-periodic linear ramp
        with no representation in any Fourier-type space (a different
        reason than :class:`PiecewiseBasis`/:class:`CubicHermiteBasis`
        raise for theirs).

        ``"sine"`` has no constant term, so its first integral is
        well-defined -- but since :math:`\\theta(a) = -\\pi`,
        :math:`\\theta(b) = \\pi` exactly, pinning the antiderivative to
        vanish at either endpoint (:meth:`FunctionSpace._integral_matrix`'s
        convention) always forces a nonzero constant term back in, so
        ``order=1`` **grows** into ``"cosine"`` at the same ``max_mode``
        rather than staying ``"sine"``. Any ``order >= 2`` raises, since
        that intermediate ``"cosine"`` result can't itself be integrated.
        """
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if order == 0:
            return self
        if self.kind in ("full", "cosine"):
            raise NotImplementedError(
                f"{type(self).__name__}(kind={self.kind!r}) does not define "
                f"an integral basis: it contains the constant/DC basis "
                f"function, whose antiderivative is a non-periodic linear "
                f"ramp with no representation in any Fourier-type space. "
                f"Project onto a non-periodic FunctionSpace (e.g. "
                f"FunctionSpace.legendre) to integrate exactly instead."
            )
        if order >= 2:
            raise NotImplementedError(
                "FourierBasis(kind='sine')'s order-1 integral is "
                "kind='cosine', which cannot itself be integrated (it "
                "contains a DC term); only order=1 is supported here."
            )
        return FourierBasis(self.max_mode + 1, kind="cosine", density=self.density)

    def evaluate(self, x, deriv: int = 0, side: str = RIGHT, **domain_kwargs):
        # `side` is validated but unused: this family is smooth everywhere,
        # so both one-sided limits agree. See `Basis.evaluate`.
        _check_side(side)
        if deriv < 0:
            raise ValueError(f"deriv must be >= 0, got {deriv}")

        scale, shift = UnitInterval().affine_params(**domain_kwargs)
        omega = np.pi / scale
        theta = omega * (x - shift)  # (npts,)
        mass = 1.0 if self.density else LegendreMeasure().mass(**domain_kwargs)
        phase = deriv * np.pi / 2
        N = self.max_mode

        parts = []
        if self.kind in ("full", "cosine"):
            const = np.ones_like(x) / np.sqrt(mass) if deriv == 0 else np.zeros_like(x)
            parts.append(const[:, None])  # (npts, 1)

        if N > 0:
            k = np.arange(1, N + 1, dtype=float)  # (N,) -- static, not traced
            factor = np.sqrt(2.0 / mass) * (k * omega) ** deriv  # (N,)
            k_theta = theta[:, None] * k[None, :]  # (npts, N), one broadcast
            cos_terms = factor * np.cos(k_theta + phase)  # (npts, N)
            sin_terms = factor * np.sin(k_theta + phase)  # (npts, N)

            if self.kind == "full":
                # Interleave cos_1, sin_1, ..., cos_N, sin_N per point via a
                # static permutation matrix rather than a 3-D stack+reshape
                # or fancy column indexing: CasADi's symbolic arrays are
                # strictly 2-D and don't support either of those, but a
                # matmul by a static (2N, 2N) 0/1 matrix is an ordinary,
                # symbolic-safe operation (like every differentiation-matrix
                # product elsewhere in this module).
                combined = np.concatenate([cos_terms, sin_terms], axis=-1)
                parts.append(combined @ _interleave_matrix(N))
            elif self.kind == "cosine":
                parts.append(cos_terms)
            else:  # "sine"
                parts.append(sin_terms)

        return np.concatenate(parts, axis=-1)
