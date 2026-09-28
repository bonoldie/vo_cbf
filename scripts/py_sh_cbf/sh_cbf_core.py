import numpy as np
from scipy import optimize


def class_K_function(h, gamma=1.0, beta=1.0):
    return gamma * h + beta * h**3


def compute_sh_parameters(ell, R, n, tau):
    """
    Computes the super-hyperbola parameters fixed by the FVO cut-off disk
    D(p/tau, R/tau), with ell = ||p|| > R.

    Returns the vertex a, the width b of the narrowest super-hyperbola that
    contains the disk, the ordinate y_star of the tangency point and db/dell.
    """
    d = ell / tau
    r = R / tau
    a = d - r

    if n == 2:
        # The hyperbola osculates the disk at the vertex
        b = np.sqrt(a * r)
        return a, b, a, b / (2.0 * a * tau)

    # Tangency polynomial P(y) deflated by its root y = a. By Lemma 1 the other
    # positive root is simple and lies in (a, d - r^2/d), where the deflated
    # polynomial goes from negative to positive.
    def deflated_tangency_poly(y):
        s = sum(y**(n - 1 - k) * a**k for k in range(n))  # (y^n - a^n) / (y - a)
        return (d + r - y) * y**(n - 1) + (y - d) * s

    y_star = optimize.brentq(deflated_tangency_poly, a, d - r**2 / d, xtol=1e-14)

    b = a * np.sqrt(r**2 - (y_star - d)**2) / (y_star**n - a**n)**(1.0 / n)
    db = b / tau * (1.0 / a - (y_star**(n - 1) - a**(n - 1)) / (y_star**n - a**n))

    return a, b, y_star, db


def compute_and_eval_h_and_grad(
    robot_state: np.ndarray,
    obstacle_state: np.ndarray,
    robot_radius: float,
    obstacle_radius: float,
    n: int,
    tau: float,
):
    """
    Computes and evaluates the CBF and its gradient w.r.t. the robot state
    [p_A, v_A] in closed form. States are [position, velocity], in 2D or 3D.

    h depends on p_B - p_A and v_A - v_B, so for an obstacle moving at
    constant velocity, with k the dimension of the space:

        L_f h = grad_h[:k] @ (v_A - v_B),    L_g h = grad_h[k:]
    """
    if n < 2 or n % 2:
        raise ValueError(f"n must be an even integer >= 2, got {n}")

    eps = 1e-12
    R = robot_radius + obstacle_radius

    robot_state = np.asarray(robot_state, dtype=float)
    obstacle_state = np.asarray(obstacle_state, dtype=float)
    k = robot_state.shape[0] // 2

    # Robot -> obstacle direction
    delta_p = obstacle_state[:k] - robot_state[:k]
    ell = np.sqrt(delta_p @ delta_p + eps**2)
    e_los = delta_p / ell

    v_rel = robot_state[k:] - obstacle_state[k:]

    # Relative velocity along and perpendicular to the line of sight
    v_parallel = v_rel @ e_los
    v_perpendicular = v_rel - v_parallel * e_los
    v_tangential_sq = v_perpendicular @ v_perpendicular

    if ell <= R:
        # Overlapping: as ell -> R, a and b vanish and the collision set tends
        # to the half-space v_parallel > 0, so h -> -v_parallel
        h_value = -v_parallel
        grad_h_value = np.concatenate((v_perpendicular / ell, -e_los))
        return 0.0, 0.0, 0.0, h_value, grad_h_value

    a, b, y_star, db = compute_sh_parameters(ell, R, n, tau)

    # n is even, so v_t^n = (v_t^2)^(n/2) and h is smooth at v_t = 0
    v_tangential_n = v_tangential_sq ** (n // 2)
    phi = a * (1.0 + v_tangential_n / b**n) ** (1.0 / n)
    gamma = phi * v_tangential_sq ** (n // 2 - 1) / (b**n + v_tangential_n)

    h_value = phi - v_parallel

    # p_A enters through ell (in a and b) and through e_los (in v_r and v_t)
    grad_p = (
        -(phi / (a * tau) - gamma * v_tangential_sq * db / b) * e_los
        + (1.0 + gamma * v_parallel) * v_perpendicular / ell
    )
    grad_v = gamma * v_perpendicular - e_los

    return b, a, y_star, h_value, np.concatenate((grad_p, grad_v))
