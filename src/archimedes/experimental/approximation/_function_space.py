"""Finite-dimensional linear span of a Basis, tied to a target domain."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from archimedes import tree
from archimedes.quadrature import Quadrature, QuadratureRule

from ._basis import Basis

if TYPE_CHECKING:
    from ._function import Function

__all__ = ["FunctionSpace"]


def _is_superset(have: np.ndarray, required: np.ndarray, tol: float = 1e-12) -> bool:
    """Whether every point of ``required`` appears in ``have``.

    A superset, not equality: refining an element is harmless, since each
    sub-element still lies inside one element of the basis. Conversely a
    finer rule that is *not* aligned is still wrong, so comparing node
    counts would prove nothing.
    """
    return bool(np.all([np.any(np.abs(have - point) <= tol) for point in required]))


@tree.struct
class FunctionSpace:
    """The linear span of a :class:`Basis` on a fixed target domain.

    Pairs a ``Basis`` (family + size) with a target domain and a "natural"
    quadrature rule, and provides the operations that act on the *space*
    rather than on any one element of it: evaluation, the mass and
    stiffness matrices, and (Galerkin) projection.

    See :class:`Function` for a specific element of the space (a
    coefficient vector).

    This is a ``@struct`` rather than a plain dataclass so that ``domain``
    is a pytree leaf: the domain parameters can be symbolically traced,
    and so optimized over (moving the endpoints of an element, say) jointly
    with a ``Function``'s coefficients. ``basis`` and ``quad_rule`` are
    static -- they carry structure, not numbers.

    Parameters
    ----------
    basis : Basis
        Basis family and size. Static.
    domain : basis.Parameters
        Target-domain parameters, validated and typed per ``basis`` -- an
        instance of ``basis.Parameters`` (e.g.
        ``UnitInterval.Parameters(a=..., b=...)`` for a basis on an
        interval, ``RealLine.Parameters(mean=..., std=...)`` for a
        Hermite-derived one). A pytree leaf, so it may be traced.
    quad_rule : QuadratureRule, optional
        The space's natural quadrature rule, used unconditionally by
        ``mass_matrix``/``stiffness_matrix`` (their integrands' required
        accuracy is fully determined by ``basis``, so there's no reason to
        let it drift from this default) and as the default for ``project``
        (which accepts an explicit override, since the right accuracy
        for a given target function isn't knowable from the space alone).
        Static.

        Defaults to ``basis.default_quadrature()``, which is exact for
        those integrands by construction. An explicit rule is checked for
        compatibility with the basis and rejected if it cannot integrate it
        exactly -- see ``Basis.required_breakpoints``.
    """

    basis: Basis = tree.field(static=True)
    domain: Any
    quad_rule: Quadrature | None = tree.field(static=True, default=None)

    def __post_init__(self):
        if not isinstance(self.domain, self.basis.Parameters):
            raise TypeError(
                f"domain must be a {self.basis.Parameters.__qualname__} "
                f"instance for this basis, got {type(self.domain).__name__}"
            )

        if self.quad_rule is None:
            object.__setattr__(self, "quad_rule", self.basis.default_quadrature())
        else:
            self._validate_quad_rule(self.quad_rule)

    def _validate_quad_rule(self, rule: Quadrature) -> None:
        """Reject a rule that cannot integrate this basis correctly.

        Three structural requirements:

        - **Dimension.** A rule of the wrong ``ndim`` presents points of the
          wrong shape.
        - **Weight.** ``scaled_weights`` applies the *rule's* weight
          implicitly, so a rule built on a different measure computes a
          different inner product than the basis is orthogonal under.
        - **Alignment.** Where the basis is not smooth, the rule's
          subintervals must not straddle the kinks.

        Degree is *not* checked; an under-resolved rule is inaccurate but
        not categorically wrong, and ``project`` legitimately varies it.
        """
        ndim = self.basis.ndim
        if rule.ndim != ndim:
            raise ValueError(
                f"{type(self.basis).__name__} is {ndim}-dimensional but the "
                f"quadrature rule is {rule.ndim}-dimensional"
            )

        # Both sides are per-dimension tuples of length `ndim`; a basis with
        # no weight of its own (nodal, piecewise) reports None and imposes
        # no constraint.
        for d, (basis_measure, rule_measure) in enumerate(
            zip(self.basis.measures, rule.measures)
        ):
            if basis_measure is not None and basis_measure != rule_measure:
                where = "" if ndim == 1 else f" in dimension {d}"
                raise ValueError(
                    f"quadrature weight does not match the basis{where}: the "
                    f"basis is orthogonal under "
                    f"{type(basis_measure).__name__} but the rule integrates "
                    f"against {type(rule_measure).__name__}, so the two "
                    f"describe different inner products"
                )

        required = self.basis.required_breakpoints
        if required is None:
            return

        # `required_breakpoints`/`breakpoints` are per-dimension tuples for a
        # tensor basis/rule and bare values otherwise.
        per_dim_required = required if ndim > 1 else (required,)
        per_dim_have = rule.breakpoints if ndim > 1 else (rule.breakpoints,)
        for d, (req, have) in enumerate(zip(per_dim_required, per_dim_have)):
            if req is None:
                continue
            if have is None or not _is_superset(have, req):
                where = "" if ndim == 1 else f" in dimension {d}"
                raise ValueError(
                    f"{type(self.basis).__name__} is only piecewise smooth"
                    f"{where}, with breakpoints {np.asarray(req)}, so "
                    f"quadrature elements must not straddle them; got a rule "
                    f"with breakpoints {have}. Use `composite(rule, "
                    f"breakpoints)` over a superset of the basis breakpoints, "
                    f"or omit `quad_rule` to use `basis.default_quadrature()`."
                )

    @property
    def n_basis(self) -> int:
        """Number of basis functions; forwarded from ``basis``."""
        return self.basis.n_basis

    def _is_compatible_with(self, other: FunctionSpace) -> bool:
        """Whether ``other`` denotes the same space, as far as is decidable.

        Compares ``basis`` and ``quad_rule`` by value, and ``domain`` only
        *structurally* (via its treedef) -- deliberately not by value.

        Domain values can't be compared once traced: two symbolic
        parametrizations raise ``TypeError`` under ``bool()`` unless they
        happen to be the same objects, and independently-traced ``Function``
        arguments get distinct symbols even when their source domains were
        numerically identical. So value comparison would be both
        undecidable and prone to false rejection.

        This means a caller can add two ``Function``s whose domains differ
        *numerically* -- e.g. ``(a=0, b=1)`` and ``(a=0, b=2)`` -- without
        an error. **Callers are responsible for ensuring the domains agree
        numerically**; only the structure is enforced here.
        """
        return (
            self.basis == other.basis
            and self.quad_rule == other.quad_rule
            and tree.structure(self.domain) == tree.structure(other.domain)
        )

    def _product_space(self, other: FunctionSpace) -> FunctionSpace:
        """The space that represents products of elements of ``self`` and
        ``other`` exactly.

        Uses ``basis._product_basis`` for the enlarged basis and this space's
        ``domain``; the quadrature rule is the product basis's own default,
        which is automatically exact for the product. With ``n_1 + n_2 - 1``
        Gauss points that rule is exact through degree
        ``2(n_1 + n_2) - 3``, and both the mass matrix and the load vector
        of the projection have degree ``2(n_1 + n_2 - 2)``.

        The domains are checked structurally only, for the same reason as
        :meth:`_is_compatible_with`: they may be traced.
        """
        if tree.structure(self.domain) != tree.structure(other.domain):
            raise ValueError(
                "product requires Functions on structurally identical domains"
            )
        return FunctionSpace(self.basis._product_basis(other.basis), domain=self.domain)

    def _derivative_space(self, deriv=1) -> FunctionSpace:
        """The smallest space that represents ``deriv``-th derivatives of
        this space's elements exactly.

        Uses the domain and derivative basis from this space's ``domain`` and
        ``basis``, with the derivative basis's own default quadrature. Smaller
        than this space for a polynomial family (differentiating lowers the degree);
        see :meth:`Basis._derivative_basis` for why the minimal space rather
        than this one.
        """
        return FunctionSpace(self.basis._derivative_basis(deriv), domain=self.domain)

    def diff_matrix(self, deriv=1, space: FunctionSpace | None = None) -> np.ndarray:
        """Matrix mapping this space's coefficients to those of the
        ``deriv``-th derivative.

        .. math::
            D = M^{-1} \\, \\Phi_t^\\top W \\, \\Phi^{(k)},

        the Galerkin projection of :math:`\\phi_i^{(k)}` onto the target
        space, with :math:`M` and the quadrature taken from that target.
        This is exact whenever the target space contains the derivative,
        which it does by construction for both defaults below.

        Parameters
        ----------
        deriv : int or tuple of int, optional
            Derivative order; a multi-index for a
            :class:`TensorBasis`, a plain order otherwise. Default 1.
        space : FunctionSpace, optional
            Target space, overriding the default. By default the target is
            **this** space, giving the square ``(n_basis, n_basis)``
            differentiation matrix -- the form collocation and operator
            assembly want, since it keeps the coefficients' meaning (nodal
            values, modal amplitudes) unchanged. Use :meth:`Function.derivative`
            for the minimal target.

            A target too small to hold the derivative gives its projection,
            which is a well-defined approximation but no longer exact.

        Returns
        -------
        ndarray
            Shape ``(space.n_basis, self.n_basis)``.
        """
        target = self if space is None else space
        rule = target.quad_rule
        _, w = target._quad_points_weights(rule)
        phi = target._basis_eval_at_nodes(rule)  # (npts, n_target)
        dphi = self._basis_eval_at_nodes(rule, deriv=deriv)  # (npts, n_basis)
        M = phi.T @ (w[:, None] * phi)
        return np.linalg.solve(M, phi.T @ (w[:, None] * dphi))  # type: ignore[no-any-return]

    def _domain_kwargs(self) -> dict:
        return {f.name: getattr(self.domain, f.name) for f in tree.fields(self.domain)}

    def _basis_eval(self, x, deriv: int = 0):
        return self.basis.evaluate(x, deriv=deriv, **self._domain_kwargs())

    def _basis_eval_at_nodes(self, rule: Quadrature, deriv: int = 0):
        """Design matrix at ``rule``'s nodes.

        Distinct from ``_basis_eval(rule.scaled_points(...))`` only for a
        basis that is discontinuous at its breakpoints, where the rule's
        record of which element each node came from resolves an ambiguity
        the coordinates cannot. See :meth:`Basis._evaluate_at_nodes`.
        """
        return self.basis._evaluate_at_nodes(rule, deriv=deriv, **self._domain_kwargs())

    def _resolve_rule(self, quad_rule: Quadrature | None = None) -> Quadrature:
        return quad_rule if quad_rule is not None else self.quad_rule

    def _quad_points_weights(self, quad_rule: Quadrature | None = None):
        rule = quad_rule if quad_rule is not None else self.quad_rule
        # phi is (npts, n_basis), so phi.T @ diag(w) @ phi has rank at most
        # min(npts, n_basis) -- below n_basis points the mass matrix is
        # exactly (not just poorly) singular, and np.linalg.solve in
        # `project` blows up rather than failing cleanly.
        if len(rule) < self.n_basis:
            raise ValueError(
                f"quad_rule has {len(rule)} points, fewer than n_basis="
                f"{self.n_basis}; the mass matrix would be singular"
            )
        domain_kwargs = self._domain_kwargs()
        # `density=self.basis.density` keeps the quadrature weights consistent
        # with the basis's own normalization (see `Basis.density`): a basis
        # orthonormal w.r.t. a probability measure needs weights that
        # integrate that same probability measure, not the raw weight.
        weights = rule.scaled_weights(**domain_kwargs, density=self.basis.density)
        return rule.scaled_points(**domain_kwargs), weights

    def evaluate(self, coefficients: np.ndarray, x, deriv: int = 0, **kwargs):
        """Evaluate :math:`\\sum_i c_i \\, \\phi_i(x)` (or its ``deriv``-th
        derivative) at ``x``, for coefficients ``c = coefficients``.

        Parameters
        ----------
        coefficients : ndarray
            Shape ``(n_basis,)`` for a scalar-valued function, or
            ``(n_basis, m)`` for an ``m``-component vector-valued one (see
            :class:`Function`).
        x : array_like
            Evaluation points, shape ``(npts,)``.
        deriv : int, optional
            Derivative order. Default 0.
        side: str, optional
            Forwarded to the basis, for options only some families accept, e.g.
            currently ``side`` (:class:`PiecewiseBasis`), selecting which
            one-sided limit to take at a breakpoint with discontinuous elements.

        Returns
        -------
        ndarray
            Shape ``(npts,)`` or ``(npts, m)``, matching ``coefficients``.
        """
        # `evaluate_expansion` rather than `_basis_eval(...) @ coefficients`:
        # a locally-supported basis can fuse the two and avoid materializing
        # the full (npts, n_basis) matrix. The default implementation is
        # exactly that matrix product.
        return self.basis.evaluate_expansion(
            coefficients, x, deriv=deriv, **self._domain_kwargs(), **kwargs
        )

    def inner_product(
        self,
        c1: np.ndarray,
        c2: np.ndarray,
        quad_rule: Quadrature | None = None,
    ):
        """Inner product :math:`\\langle f, g \\rangle = \\int f(x) \\, g(x)
        \\, w(x) \\, dx` for ``f``, ``g`` in this space with coefficients
        ``c1``, ``c2``, approximated via ``quad_rule`` (default
        ``self.quad_rule``).

        Unlike a product of two ``Function``s (deliberately unsupported --
        see :class:`Function`), an inner product returns a scalar rather
        than another element of the space, so there's no aliasing/closure
        question to resolve: it's computed by evaluating both functions at
        the quadrature nodes and integrating the pointwise product, which
        is exact whenever ``quad_rule`` is accurate enough for that
        product -- equivalently ``c1 @ mass_matrix() @ c2``, but computed
        directly without forming the full ``(n_basis, n_basis)`` matrix.

        For vector-valued coefficients (shape ``(n_basis, m)``) the
        integrand is contracted over components, :math:`\\langle f, g
        \\rangle = \\int f \\cdot g \\, w \\, dx`, so the result is a
        scalar in that case too and :meth:`Function.norm` is the
        :math:`L^2` norm of the whole vector-valued function rather than
        an array of per-component norms. Both coefficient arrays must have
        the same shape; for a per-component inner product, slice the
        coefficients and call this once per component.
        """
        rule = self._resolve_rule(quad_rule)
        _, w = self._quad_points_weights(rule)
        phi = self._basis_eval_at_nodes(rule)  # (npts, n_basis)
        integrand = (phi @ c1) * (phi @ c2)  # (npts,) or (npts, m)
        if integrand.ndim > 1:
            # `ndim` is a static (trace-time) property, so this branches on
            # shape rather than on a value and is safe under `@arc.compile`.
            integrand = np.sum(integrand, axis=-1)
        return np.dot(w, integrand)

    def mass_matrix(self) -> np.ndarray:
        """Mass matrix :math:`M_{ij} = \\int \\phi_i \\, \\phi_j \\, w \\, dx`,
        approximated via ``self.quad_rule``.
        """
        _, w = self._quad_points_weights()
        phi = self._basis_eval_at_nodes(self.quad_rule)  # (npts, n_basis)
        return phi.T @ (w[:, None] * phi)

    def stiffness_matrix(self) -> np.ndarray:
        """Stiffness matrix :math:`K_{ij} = \\int \\nabla \\phi_i \\cdot
        \\nabla \\phi_j \\, w \\, dx`, approximated via ``self.quad_rule``.

        In one dimension this is the usual :math:`\\int \\phi_i' \\phi_j'`.
        In more, "the derivative" is ambiguous, and the form that appears in
        a Laplacian is the gradient one -- a sum over dimensions of the
        per-direction stiffness, using the unit multi-indices as ``deriv``.
        """
        _, w = self._quad_points_weights()
        ndim = self.basis.ndim
        if ndim == 1:
            derivs: list = [1]
        else:
            derivs = [
                tuple(1 if d == k else 0 for d in range(ndim)) for k in range(ndim)
            ]

        stiffness = None
        for deriv in derivs:
            dphi = self._basis_eval_at_nodes(self.quad_rule, deriv=deriv)
            block = dphi.T @ (w[:, None] * dphi)
            stiffness = block if stiffness is None else stiffness + block
        return stiffness

    def project(self, f: Callable, quad_rule: Quadrature | None = None) -> Function:
        """Galerkin projection of ``f`` onto this space.

        Solves ``M @ c = b`` for the coefficients ``c``, where ``M`` is the
        mass matrix and ``b_i = int f(x) phi_i(x) w(x) dx``, both
        approximated via ``quad_rule`` (default ``self.quad_rule``).
        ``quad_rule`` must be accurate enough for the product of ``f`` and
        the basis, which is generally a higher-order requirement than
        exactness for the basis alone; pass an explicit ``quad_rule`` to
        use something other than the space's natural default.

        ``f`` may be vector-valued: if ``f(x)`` has shape ``(npts, m)``,
        each component is projected onto the same space and the result has
        coefficients of shape ``(n_basis, m)``. The mass matrix is shared
        across components, so this costs one basis evaluation rather than
        ``m`` of them.

        Parameters
        ----------
        f : callable
            Target function, called once as ``f(x)`` on the full node
            array from the quadrature rule. Must return an array of shape
            ``(npts,)`` (scalar-valued) or ``(npts, m)`` (vector-valued).
        quad_rule : QuadratureRule, optional
            Quadrature rule to use instead of ``self.quad_rule``.

        Returns
        -------
        Function
            The projected function, in this space, with coefficients of
            shape ``(n_basis,)`` or ``(n_basis, m)`` to match ``f``.
        """
        from ._function import Function  # avoid a circular import

        rule = self._resolve_rule(quad_rule)
        x, w = self._quad_points_weights(rule)
        phi = self._basis_eval_at_nodes(rule)  # (npts, n_basis)
        M = phi.T @ (w[:, None] * phi)
        # `f` is an ordinary function of position, so it needs the coordinates
        # and has no breakpoint ambiguity of its own to resolve.
        fx = f(x)

        # `ndim` is a static (trace-time) property, so this branches on shape
        # rather than on a value and is safe under `@arc.compile`. In the
        # vector-valued case the right-hand side is the (n_basis, m) matrix of
        # stacked component loads, which `solve` handles with a single
        # factorization of the shared mass matrix.
        rhs = phi.T @ (w * fx if fx.ndim == 1 else w[:, None] * fx)
        return Function(np.linalg.solve(M, rhs), self)
