"""hp-adaptive mesh refinement based on Darby et al (2011)

https://people.clas.ufl.edu/hager/files/RaoDarbyGridRefineJournal.pdf
"""

import numpy as np

from .discretization import RadauFiniteElements


def midpoint_residuals(ocp, domain, sol):
    # Differentiate then interpolate
    idx = 0
    r = []
    for k in range(domain.n_elements):
        element = domain.elements[k]
        N = element.n_nodes
        xp = sol.xp[idx : idx + N + 1]
        tp = sol.tp[idx : idx + N + 1]
        idx += N

        res_fn = ocp.dynamics_residual(sol, element, xp, tp[0], tp[-1])

        # Midpoint residuals
        t_mid = 0.5 * (tp[1:-1] + tp[:-2])
        R = res_fn(t_mid)
        r.append(np.max(R, axis=1))  # Maximum element in each row

    return r


def refine_mesh_bisection(
    domain, residuals, eps=1e-4, incr=10, max_nodes=50, rho=3, verbose=False
):
    next_N = []
    next_knots = []
    all_converged = True
    for k, r in enumerate(residuals):
        element = domain.elements[k]
        N = element.n_nodes
        τ = domain.local_to_global(k)
        τ_mid = 0.5 * (τ[0] + τ[-1])  # Global midpoints

        beta = r / np.mean(r)  # Scaled residual

        if verbose:
            print(
                f"Element {k}: residual (max/avg) = {np.max(r):.2e}/{np.mean(r):.2e}, max scaled = {np.max(beta):.2e}"
            )

        # Check for convergence (all residuals < eps)
        if np.all(r < eps):
            next_N.append(N)

            if verbose:
                print("\tConverged")

        else:
            all_converged = False

            # Split at the midpoints if there are too many nodes or
            # non-uniform errors
            if (N + incr > max_nodes) or np.any(beta > rho):
                next_N.extend([incr, incr])
                next_knots.append(τ_mid)

                if verbose:
                    print(f"\tSplitting at τ={τ_mid}")

            # Refine element
            else:
                next_N.append(N + incr)

        if k < domain.n_elements - 1:
            next_knots.append(domain.knots[k + 1])

    if verbose:
        print(f"New domain: {next_N}, knots={next_knots}")

    next_domain = RadauFiniteElements(N=next_N, knots=next_knots)
    return all_converged, next_domain


# TODO: Fix or remove
# def refine_mesh_localized(domain, residuals, eps=1e-4, rho=3, L=10, verbose=False):
#     next_N = []
#     next_knots = []
#     all_converged = True
#     for k, r in enumerate(residuals):
#         element = domain.elements[k]
#         τ = domain.local_to_global(k)
#         τ_mid = 0.5 * (τ[1:-1] + τ[:-2])  # Global midpoints

#         N = element.n_nodes
#         beta = r / np.mean(r)  # Scaled residual

#         if verbose:
#             print(f"Element {k}: residual (max/avg) = {np.max(r):.2e}/{np.mean(r):.2e}, max scaled = {np.max(beta):.2e}")

#         # Check for convergence (all beta < eps)
#         if np.all(r < eps):
#             uniform = True
#             converged = True

#             if verbose:
#                 print("\tConverged")

#         # Check if errors are uniform (all beta < rho)
#         elif np.all(beta < rho):
#             all_converged = False
#             converged = False
#             uniform = True

#             if verbose:
#                 print(f"\tUniform errors. Refining to N={N + L}")

#         # Nonuniform errors: find location of segment breaks
#         else:
#             all_converged = False
#             converged = False
#             uniform = False
#             split_idx = np.where(beta > rho)[0]

#             if verbose:
#                 print(f"\tSplitting at τ={τ_mid[split_idx]}")

#         if uniform:
#             next_N.append(N if converged else N + L)

#         else:
#             next_N.extend([L] * (len(split_idx) + 1))
#             next_knots.extend(τ_mid[split_idx])

#         if k < domain.n_elements - 1:
#             next_knots.append(domain.knots[k+1])

#     if verbose:
#         print(f"New domain: {next_N}, knots={next_knots}")

#     next_domain = RadauFiniteElements(N=next_N, knots=next_knots)
#     return all_converged, next_domain
