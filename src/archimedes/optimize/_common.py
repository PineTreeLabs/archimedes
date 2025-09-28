from archimedes import tree

__all__ = ["_ravel_args"]


def _ravel_args(x, bounds, zip_bounds=False):
    # By default, just flatten/unflatten the tree
    x_flat, unravel_x = tree.ravel(x)

    if bounds is not None:
        lower, upper = bounds
        lb, _ = tree.ravel(lower)
        ub, _ = tree.ravel(upper)
        x_treedef = tree.structure(x)
        lb_treedef = tree.structure(lower)
        ub_treedef = tree.structure(upper)
        if x_treedef != lb_treedef:
            raise ValueError(
                f"Lower bounds must have the same structure as x0 but got {x_treedef} "
                f"for x0 and {lb_treedef} for lower bounds."
            )
        if len(lb) != len(x_flat):
            raise ValueError(
                f"Lower bounds must have the same number of elements as x0 but got "
                f"{len(lb)} for lower bounds and {len(x_flat)} for x0."
            )
        if x_treedef != ub_treedef:
            raise ValueError(
                f"Upper bounds must have the same structure as x0 but got {x_treedef} "
                f"for x0 and {ub_treedef} for upper bounds."
            )
        if len(ub) != len(x_flat):
            raise ValueError(
                f"Upper bounds must have the same number of elements as x0 but got "
                f"{len(ub)} for upper bounds and {len(x_flat)} for x0."
            )

        if zip_bounds:
            # Zip bounds into (lb, ub) for each parameter (for SciPy compatibility)
            bounds_flat = list(zip(lb, ub))
        else:
            bounds_flat = (lb, ub)  # type: ignore[assignment]

    else:
        bounds_flat = None

    return x_flat, bounds_flat, unravel_x
