"""Test-only re-creations of ``FunctionSpace.mass_matrix``/``.stiffness_matrix``.

Both were removed from the public API: they had zero consumers beyond these
tests (confirmed by grep across src/test/docs), and are now one-off
conveniences a user reconstructs from ``design_matrix``/``quadrature`` and
``archimedes.quadrature.contract`` in a couple of lines, rather than the
library shipping and maintaining them. These recreate them for test purposes
only, to keep the correctness coverage they provided without reintroducing
the methods themselves -- exactly the pattern a power user would follow.
"""

from archimedes.quadrature import contract


def mass_matrix(space):
    _, w = space.quadrature()
    phi = space.design_matrix()
    return contract(phi, w, phi)


def stiffness_matrix(space):
    _, w = space.quadrature()
    ndim = space.basis.ndim
    if ndim == 1:
        derivs = [1]
    else:
        derivs = [tuple(1 if d == k else 0 for d in range(ndim)) for k in range(ndim)]

    stiffness = None
    for deriv in derivs:
        dphi = space.design_matrix(deriv=deriv)
        block = contract(dphi, w, dphi)
        stiffness = block if stiffness is None else stiffness + block
    return stiffness
