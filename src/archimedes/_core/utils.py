import casadi as cs

import archimedes as arc
from archimedes._core._array_impl import SymbolicArray, _as_casadi_array


@arc.compile
def find_equal(x, xp, yp):
    # Return the first value of yp[j] such that xp[j] >= x.

    # Since this is a compiled function, we can assume that both are symbolic arrays
    xp_cs = _as_casadi_array(xp)
    x_cs = _as_casadi_array(x)

    # Add a dummy value or CasADi will only go to the second-to-last element
    inf_ = cs.MX.inf()  # low only supports MX
    grid = cs.vcat([xp_cs, inf_])
    i_cs = cs.low(grid, x_cs)
    y_cs = yp._sym[i_cs, :]

    # FIXME: This could be a more general function, but it would need much
    # more careful shape checking.
    return SymbolicArray(y_cs, shape=yp[0].shape, dtype=yp.dtype)
