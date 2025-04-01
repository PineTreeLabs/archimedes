import numpy as np

from archimedes import compile, jac, odeint


def lqr_design(f, x0, u0, ts, Qf, Q, R):
    """Construct a finite-horizon LQR control law

    Args:
        f(t, x, u): dynamics function
        x0: nominal state (or function of time)
        u0: nominal control (or function of time)
        ts: time points for RDE solve
        Qf: final cost matrix
        Q: cost matrix
        R: control cost matrix
    """
    nx = Q.shape[0]
    nu = R.shape[0]
    R_inv = np.linalg.inv(R)

    if isinstance(x0, np.ndarray):

        def _x0(t):
            return x0
    else:
        _x0 = x0

    if isinstance(u0, np.ndarray):

        def _u0(t):
            return u0
    else:
        _u0 = u0

    # Linearize the dynamics about the nominal trajectory
    dfdx = jac(f, argnums=1)
    dfdu = jac(f, argnums=2)

    def linearize(t):
        x0, u0 = _x0(t), _u0(t)
        A = dfdx(t, x0, u0)
        B = dfdu(t, x0, u0)
        return A, B

    # Differential Riccatti equation (square root form)
    # Integrate Riccati equation from t = tf to t = 0
    # Transform time variable to tau = -t and
    # integrate from tau = -t0 to tau = -t0
    # In this case we just change the sign of S_t (usually has a negative sign
    # when integrating backwards in time)
    @compile(kind="MX")
    def riccati_rhs(t, x):
        t = -t  # Transform from tau=-t to t
        A, B = linearize(t)
        P = np.reshape(x, (nx, nx))
        S = P @ P.T
        P_t = A.T @ P - 0.5 * S @ (B @ R_inv @ B.T) @ P + 0.5 * Q * np.linalg.inv(P)
        return P_t.flatten()

    P0 = np.linalg.cholesky(Qf)
    t0, tf = ts[0], ts[-1]
    P = odeint(riccati_rhs, x0=P0.flatten(), t_span=(-tf, -t0), t_eval=ts)
    P = np.reshape(P.T, (len(ts), nx, nx))
    S = np.array([P[i] @ P[i].T for i in range(len(ts))])

    Ks = np.zeros((len(ts), nu, nx))
    for i in range(len(ts)):
        _A, B = linearize(ts[i])
        Ks[i] = R_inv @ B.T @ S[i]

    def K(t):
        # TODO: Use barycentric interpolation here?
        # Or something that doesn't require repeated interpolation
        _K = [np.interp(t, ts, Ks[:, i, j]) for i in range(nu) for j in range(nx)]
        return np.stack(_K).reshape((nu, nx))

    def u_lqr(t, x):
        return -K(t) @ (x - _x0(t))

    return u_lqr
