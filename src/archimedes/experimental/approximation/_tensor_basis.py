"""Tensor-product basis: one univariate Basis per dimension."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from archimedes import tree
from archimedes.measure import Measure, ReferenceDomain

from ._basis import Basis

__all__ = ["ProductParameters", "TensorBasis"]


@tree.struct
class ProductParameters(ReferenceDomain.Parameters):
    """Target-domain parameters for a :class:`TensorBasis`: one
    :class:`~archimedes.measure.ReferenceDomain.Parameters` per dimension.

    A single nested field rather than one flattened set of parameters per
    dimension, because the names would otherwise collide -- two interval
    dimensions both want ``a``/``b``. Keeping them nested also means
    ``FunctionSpace``'s existing ``**domain_kwargs`` convention carries
    them through unchanged, as the single keyword ``dims``.

    The entries are ordinary ``@struct`` types, so their fields remain
    pytree leaves: a tensor domain can be traced and optimized over exactly
    like a one-dimensional one.

    Parameters
    ----------
    dims : tuple of ReferenceDomain.Parameters
        One entry per dimension, in the same order as the basis's factors.
    """

    dims: tuple = ()

    def __post_init__(self):
        dims = tuple(self.dims)
        if not dims:
            raise ValueError("ProductParameters needs at least one dimension")
        for i, spec in enumerate(dims):
            if not isinstance(spec, ReferenceDomain.Parameters):
                raise TypeError(
                    f"dims[{i}] must be a ReferenceDomain.Parameters, got "
                    f"{type(spec).__name__}"
                )
        object.__setattr__(self, "dims", dims)


# TODO: Is this redundant with quadrature._tensor._dim_args?
def _dim_kwargs(basis: Basis, spec: Any) -> dict:
    """Normalize one dimension's domain parameters into keyword arguments
    for that dimension's ``basis.evaluate``.

    Accepts the reference domain (``None``), a ``ReferenceDomain.Parameters``
    struct (what a ``FunctionSpace`` domain holds), a kwargs dict, or a
    positional tuple matched against ``basis.Parameters``' field order --
    the same set of forms ``TensorQuadratureRule`` accepts, so the two
    ``dims`` conventions stay identical.
    """
    if spec is None:
        return {}
    if isinstance(spec, ReferenceDomain.Parameters):
        return {f.name: getattr(spec, f.name) for f in tree.fields(spec)}  # type: ignore[arg-type]
    if isinstance(spec, dict):
        return dict(spec)
    if isinstance(spec, (tuple, list)):
        names = [f.name for f in dataclasses.fields(basis.Parameters)]
        if len(spec) > len(names):
            raise ValueError(
                f"got {len(spec)} positional domain parameters for "
                f"{basis.Parameters.__qualname__}, which takes {len(names)}"
            )
        return dict(zip(names, spec))
    raise TypeError(
        f"per-dimension parameters must be None, a ReferenceDomain.Parameters, "
        f"a tuple of positional arguments, or a dict of keyword arguments; "
        f"got {type(spec).__name__}"
    )


class _DimensionView:
    """One dimension of a tensor rule, presented as a 1-D quadrature rule.

    A tensor rule's ``nodes``/``elements`` are ``(n, ndim)``; a univariate
    factor needs column ``d`` of each, at the *full* node count (every
    combination), not the ``n_d`` of the underlying per-dimension rule. So
    this is a view of the expanded arrays rather than ``rule.rules[d]``.

    Only the members :meth:`Basis._evaluate_at_nodes` implementations touch
    are provided, which is why this is a plain adapter and not a
    ``Quadrature``: weights are meaningless here (they do not factor per
    dimension row-wise), and nothing downstream asks for them.
    """

    def __init__(self, rule, dim: int):
        self._rule = rule
        self._dim = dim

    @property
    def breakpoints(self):
        return self._rule.breakpoints[self._dim]

    @property
    def elements(self):
        elements = self._rule.elements
        return None if elements is None else elements[:, self._dim]

    def scaled_points(self, **domain_kwargs):
        return self._rule.scaled_points(dims=self._rule_dims(**domain_kwargs))[
            :, self._dim
        ]

    def _rule_dims(self, **domain_kwargs):
        """This dimension's parameters in the slot the tensor rule expects,
        with the others left at their reference domains -- only column
        ``self._dim`` of the result is ever read."""
        dims: list = [None] * self._rule.ndim
        dims[self._dim] = domain_kwargs
        return dims


def _row_kron(mats: list) -> np.ndarray:
    """Row-wise Kronecker (Khatri-Rao) product of design matrices.

    Given ``(npts, p)`` and ``(npts, q)``, returns ``(npts, p * q)`` with
    ``out[:, i * q + j] = a[:, i] * b[:, j]`` -- i.e. the first factor
    varies slowest, matching C order and
    ``TensorQuadratureRule``'s node ordering.

    Built column by column rather than as ``a[:, :, None] * b[:, None, :]``
    because ``SymbolicArray`` supports no more than two dimensions, so the
    broadcasting form is unavailable under tracing.
    """
    out = mats[0]
    for phi in mats[1:]:
        p, q = np.shape(out)[1], np.shape(phi)[1]
        cols = [out[:, i] * phi[:, j] for i in range(p) for j in range(q)]
        out = np.stack(cols, axis=-1)
    return out


@dataclasses.dataclass(frozen=True)
class TensorBasis(Basis):
    """Tensor product of univariate bases, one per dimension.

    The basis functions are all products of one factor from each dimension,

    .. math::
        \\Phi_{(i_1, \\ldots, i_d)}(x) = \\phi^{(1)}_{i_1}(x_1) \\cdots
            \\phi^{(d)}_{i_d}(x_d),

    so ``n_basis`` is the product of the factors' sizes and a ``Function``
    on this basis spans the full ``(n_1, ..., n_d)`` coefficient array.

    The multi-index is flattened in **C order** -- last dimension varying
    fastest, i.e. ``np.ravel_multi_index``'s default -- matching
    :class:`~archimedes.quadrature.TensorQuadratureRule`'s node ordering.
    So ``coefficients.reshape(n_1, ..., n_d)`` recovers the natural array
    layout.

    **Derivatives are multi-indices.** In more than one dimension "the
    derivative" is ambiguous, so ``deriv`` is a tuple giving the order in
    each variable: ``(1, 0)`` is :math:`\\partial_x`, ``(1, 1)`` is
    :math:`\\partial_x \\partial_y`. The scalar ``0`` is accepted as shorthand
    for no derivative at all; any other integer is rejected.

    **Factors must be univariate.** Tensor products are associative, so
    nesting adds no expressive power; write ``TensorBasis((a, b, c))``
    rather than ``TensorBasis((a, TensorBasis((b, c))))``.

    Parameters
    ----------
    bases : tuple of Basis
        One univariate basis per dimension, in order. The families may
        differ, and so may their reference domains. All factors must agree on
        ``density``, since quadrature weights are normalized (or not) for
        the product measure as a whole.

    See Also
    --------
    archimedes.quadrature.tensor : The matching quadrature construction.
    """

    bases: tuple[Basis, ...]

    def __post_init__(self):
        bases = tuple(self.bases)
        if len(bases) < 1:
            raise ValueError("a tensor basis needs at least one dimension")
        for i, basis in enumerate(bases):
            if not isinstance(basis, Basis):
                raise TypeError(
                    f"bases[{i}] must be a Basis, got {type(basis).__name__}"
                )
            if basis.ndim != 1:
                raise ValueError(
                    f"bases[{i}] is {basis.ndim}-dimensional; tensor factors "
                    f"must be univariate (tensor products are associative, so "
                    f"flatten rather than nest)"
                )
        if len({basis.density for basis in bases}) > 1:
            raise ValueError(
                "all tensor factors must agree on `density`, since quadrature "
                "weights are normalized for the product measure as a whole; got "
                f"{[basis.density for basis in bases]}"
            )
        object.__setattr__(self, "bases", bases)

    @property
    def ndim(self) -> int:
        """Number of dimensions, i.e. the number of factors."""
        return len(self.bases)

    @property
    def n_basis(self) -> int:
        """Product of the factors' sizes."""
        return int(np.prod([basis.n_basis for basis in self.bases]))

    @property
    def shape(self) -> tuple[int, ...]:
        """Per-dimension sizes, so ``coefficients.reshape(basis.shape)``
        gives the natural multi-index array."""
        return tuple(basis.n_basis for basis in self.bases)

    @property
    def density(self) -> bool:
        """The factors' common ``density``; see :attr:`Basis.density`."""
        return self.bases[0].density

    @property
    def measures(self) -> tuple[Measure | None, ...]:
        """The per-dimension orthogonality weights; see
        :attr:`Basis.measures`. There is no single weight -- the dimensions
        are independent and may use different families."""
        return sum((basis.measures for basis in self.bases), ())

    @property
    def Parameters(self) -> type:  # noqa: N802
        """:class:`ProductParameters`, holding one dimension's parameters
        per factor."""
        return ProductParameters

    @property
    def required_breakpoints(self) -> tuple:
        """Per-dimension breakpoints, one entry per dimension, each ``None``
        unless that factor is only piecewise smooth.

        Always a length-``ndim`` tuple rather than a bare ``None``, so a
        consumer can zip it against a
        :class:`~archimedes.quadrature.TensorQuadratureRule`'s own
        per-dimension breakpoints.
        """
        return tuple(basis.required_breakpoints for basis in self.bases)

    def default_quadrature(self):
        """The tensor product of the factors' own default rules, which is
        therefore exact for this basis's mass and stiffness integrands in
        each variable separately."""
        from archimedes.quadrature import tensor

        return tensor(*[basis.default_quadrature() for basis in self.bases])

    def _product_basis(self, other):
        """Tensor of the factors' product bases.

        A product of tensor-product functions factorizes dimension by
        dimension, so the enlarged space does too -- there is no coupling
        across dimensions to resolve.
        """
        if not isinstance(other, TensorBasis):
            raise ValueError(
                f"cannot form a product basis between "
                f"{type(self).__name__} and {type(other).__name__}"
            )
        if self.ndim != other.ndim:
            raise ValueError(
                f"product requires the same number of dimensions, got "
                f"{self.ndim} and {other.ndim}"
            )
        return TensorBasis(
            tuple(a._product_basis(b) for a, b in zip(self.bases, other.bases))
        )

    def _derivative_basis(self, deriv=1):
        """Tensor of the factors' derivative bases.

        ``deriv`` is a multi-index, as everywhere else here, so each factor
        is differentiated to its own order and shrinks independently --
        there is no coupling across dimensions.
        """
        alpha = self._multi_index(deriv)
        derived = []
        for d, (basis, order) in enumerate(zip(self.bases, alpha)):
            try:
                derived.append(basis._derivative_basis(order))
            except ValueError as exc:
                # Name the dimension; the factor only knows its own size.
                raise ValueError(f"in dimension {d}: {exc}") from exc
        return TensorBasis(tuple(derived))

    def _multi_index(self, deriv) -> tuple[int, ...]:
        """Normalize ``deriv`` into a length-``ndim`` multi-index."""
        if isinstance(deriv, (int, np.integer)):
            if deriv == 0:
                return (0,) * self.ndim
            raise ValueError(
                f"deriv must be a multi-index (one order per dimension) for a "
                f"{self.ndim}-dimensional basis, got {deriv}; use e.g. "
                f"{(1,) + (0,) * (self.ndim - 1)} for the first partial "
                f"derivative. Only the scalar 0 is accepted as shorthand."
            )
        alpha = tuple(int(order) for order in deriv)
        if len(alpha) != self.ndim:
            raise ValueError(
                f"deriv must have one entry per dimension, got {len(alpha)} "
                f"for a {self.ndim}-dimensional basis"
            )
        if any(order < 0 for order in alpha):
            raise ValueError(f"deriv orders must be >= 0, got {alpha}")
        return alpha

    def _dim_specs(self, dims) -> tuple:
        if dims is None:
            return (None,) * self.ndim
        dims = tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(
                f"expected {self.ndim} per-dimension parameters, got {len(dims)}"
            )
        return dims

    def evaluate(self, x, deriv=0, dims=None):
        """Evaluate all ``n_basis`` product functions at ``x``.

        Parameters
        ----------
        x : array_like
            Evaluation points, shape ``(npts, ndim)`` -- one row per point,
            one column per dimension, matching
            :meth:`TensorQuadratureRule.scaled_points`.
        deriv : tuple of int, optional
            Multi-index of derivative orders, one per dimension. The scalar
            ``0`` (the default) means no derivative.
        dims : sequence, optional
            Per-dimension target-domain parameters; see :func:`_dim_kwargs`
            for the accepted forms. Omitted, every dimension uses its
            reference domain.

        Returns
        -------
        phi : ndarray
            Shape ``(npts, n_basis)``, with the multi-index flattened in C
            order.
        """
        alpha = self._multi_index(deriv)
        specs = self._dim_specs(dims)

        shape = np.shape(x)
        if len(shape) != 2 or shape[1] != self.ndim:
            raise ValueError(
                f"x must have shape (npts, {self.ndim}) for a {self.ndim}-"
                f"dimensional basis, got {shape}"
            )

        return _row_kron(
            [
                basis.evaluate(x[:, d], deriv=alpha[d], **_dim_kwargs(basis, spec))
                for d, (basis, spec) in enumerate(zip(self.bases, specs))
            ]
        )

    def _evaluate_at_nodes(self, rule, deriv=0, dims=None):
        """Per-dimension evaluation at the rule's nodes, so a factor that
        needs element provenance (a :class:`PiecewiseBasis`) gets its own
        dimension's slice of it.

        ``rule.elements`` is ``(n, ndim)`` alongside ``rule.nodes``, so each
        factor is handed a one-dimensional view of both.
        """
        alpha = self._multi_index(deriv)
        specs = self._dim_specs(dims)
        return _row_kron(
            [
                basis._evaluate_at_nodes(
                    _DimensionView(rule, d), deriv=alpha[d], **_dim_kwargs(basis, spec)
                )
                for d, (basis, spec) in enumerate(zip(self.bases, specs))
            ]
        )
