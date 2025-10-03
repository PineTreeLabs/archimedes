import numpy as np

__all__ = ["wind_frame"]


def wind_frame(v_rel_B):
    """Compute total velocity, angle of attack, and sideslip angle

    The input should be the vehicle wind-relative velocity computed in
    body-frame axes.  If the inertial velocity of the vehicle expressed in
    body-frame axes is v_B and the Earth-relative wind velocity is w_N,
    then the relative velocity is v_rel_B = v_B + R_BN @ w_N, where R_BN
    is the rotation matrix from inertial to body frame.

    If there is no wind, then v_rel_B = v_B.
    """
    u, v, w = v_rel_B
    vt = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / vt)
    return vt, alpha, beta
