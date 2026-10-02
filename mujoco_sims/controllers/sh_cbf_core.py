import numpy as np


def class_K_function(h, gamma=1.0, beta=1.0):
    return gamma * h + beta * h**3


def compute_sh_parameters_batch(ell, R, n, tau):
    """
    Computes the super-hyperbola parameters fixed by the FVO cut-off disks
    D(p/tau, R/tau), for arrays of distances ell > R and radii R.

    Returns the vertices a, the widths b of the narrowest super-hyperbolas that
    contain the disks, the ordinates y_star of the tangency points and db/dell.
    """
    ell = np.asarray(ell, dtype=float)
    R = np.broadcast_to(np.asarray(R, dtype=float), ell.shape)

    d = ell / tau
    r = R / tau
    a = d - r

    if n == 2:
        # The hyperbola osculates the disk at the vertex
        b = np.sqrt(a * r)
        return a, b, a.copy(), b / (2.0 * a * tau)

    # In the coordinates normalised by d the disk has centre 1 and radius
    # rho = R / ell, and the vertex is a_n = 1 - rho
    rho = R / ell
    a_n = 1.0 - rho

    # Tangency polynomial P(u) = u^n - (1 - rho^2) u^(n-1) - a_n^n u + a_n^n,
    # highest power first, deflated by its root u = a_n with synthetic division
    p = np.zeros((n + 1,) + ell.shape)
    p[0] = 1.0
    p[1] = -(1.0 - rho**2)
    p[n - 1] = -a_n**n
    p[n] = a_n**n

    q = np.empty((n,) + ell.shape)
    q[0] = p[0]
    for j in range(1, n):
        q[j] = p[j] + a_n * q[j - 1]

    # By Lemma 1 the other positive root is simple and lies in (a_n, 1 - rho^2),
    # where the deflated polynomial goes from negative to positive: Newton's
    # method, with a bisection step whenever it leaves the bracket
    lo = a_n.copy()
    hi = 1.0 - rho**2
    u = 0.5 * (lo + hi)

    for _ in range(100):
        # Deflated polynomial and its derivative, with Horner's scheme
        value = q[0].copy()
        slope = np.zeros_like(u)
        for j in range(1, n):
            slope = slope * u + value
            value = value * u + q[j]

        below = value < 0.0
        lo = np.where(below, u, lo)
        hi = np.where(below, hi, u)

        # Once converged, u is an end of the bracket and so is the Newton step
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = u - value / slope
        inside = (newton >= lo) & (newton <= hi)
        u_next = np.where(inside, newton, 0.5 * (lo + hi))

        converged = np.abs(u_next - u) <= 4.0 * np.finfo(float).eps * u
        u = u_next
        if np.all(converged):
            break

    y_star = d * u

    b = a * np.sqrt(r**2 - (y_star - d)**2) / (y_star**n - a**n)**(1.0 / n)
    db = b / tau * (1.0 / a - (y_star**(n - 1) - a**(n - 1)) / (y_star**n - a**n))

    return a, b, y_star, db


def compute_sh_parameters(ell, R, n, tau):
    """
    Computes the super-hyperbola parameters fixed by the FVO cut-off disk
    D(p/tau, R/tau), with ell = ||p|| > R.

    Returns the vertex a, the width b of the narrowest super-hyperbola that
    contains the disk, the ordinate y_star of the tangency point and db/dell.
    """
    a, b, y_star, db = compute_sh_parameters_batch(np.array([ell], dtype=float), R, n, tau)
    return float(a[0]), float(b[0]), float(y_star[0]), float(db[0])


def compute_and_eval_h_and_grad_batch(
    robot_state: np.ndarray,
    obstacle_states: np.ndarray,
    robot_radius: float,
    obstacle_radii,
    n: int,
    tau: float,
):
    """
    Computes and evaluates the CBFs of M obstacles, with states obstacle_states
    (M x 2k) and radii obstacle_radii (M or a scalar), and their gradients
    w.r.t. the robot state [p_A, v_A] in closed form. States are
    [position, velocity], in 2D or 3D.

    Returns b, a, y_star and h (M) and the gradients (M x 2k). With k the
    dimension of the space, for obstacles moving at constant velocity:

        L_f h = grad_h[:, :k] (v_A - v_B),    L_g h = grad_h[:, k:]
    """
    if n < 2 or n % 2:
        raise ValueError(f"n must be an even integer >= 2, got {n}")

    eps = 1e-12

    robot_state = np.asarray(robot_state, dtype=float)
    obstacle_states = np.atleast_2d(np.asarray(obstacle_states, dtype=float))
    k = robot_state.shape[0] // 2
    R = robot_radius + np.broadcast_to(np.asarray(obstacle_radii, dtype=float), obstacle_states.shape[:1])

    # Robot -> obstacle directions
    delta_p = obstacle_states[:, :k] - robot_state[:k]
    ell = np.sqrt(np.sum(delta_p**2, axis=1) + eps**2)
    e_los = delta_p / ell[:, None]

    v_rel = robot_state[k:] - obstacle_states[:, k:]

    # Relative velocities along and perpendicular to the lines of sight
    v_parallel = np.sum(v_rel * e_los, axis=1)
    v_perpendicular = v_rel - v_parallel[:, None] * e_los
    v_tangential_sq = np.sum(v_perpendicular**2, axis=1)

    # Overlapping: as ell -> R, a and b vanish and the collision set tends to
    # the half-space v_parallel > 0, so h -> -v_parallel
    a = np.zeros_like(ell)
    b = np.zeros_like(ell)
    y_star = np.zeros_like(ell)
    h_value = -v_parallel
    grad_p = v_perpendicular / ell[:, None]
    grad_v = -e_los

    apart = ell > R
    if np.any(apart):
        a_s, b_s, y_s, db_s = compute_sh_parameters_batch(ell[apart], R[apart], n, tau)

        # n is even, so v_t^n = (v_t^2)^(n/2) and h is smooth at v_t = 0
        v_t_sq = v_tangential_sq[apart]
        v_t_n = v_t_sq ** (n // 2)
        phi = a_s * (1.0 + v_t_n / b_s**n) ** (1.0 / n)
        gamma = phi * v_t_sq ** (n // 2 - 1) / (b_s**n + v_t_n)

        a[apart], b[apart], y_star[apart] = a_s, b_s, y_s
        h_value[apart] = phi - v_parallel[apart]

        # p_A enters through ell (in a and b) and through e_los (in v_r and v_t)
        grad_p[apart] = (
            -(phi / (a_s * tau) - gamma * v_t_sq * db_s / b_s)[:, None] * e_los[apart]
            + (1.0 + gamma * v_parallel[apart])[:, None] * v_perpendicular[apart] / ell[apart, None]
        )
        grad_v[apart] = gamma[:, None] * v_perpendicular[apart] - e_los[apart]

    return b, a, y_star, h_value, np.concatenate((grad_p, grad_v), axis=1)


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
    b, a, y_star, h_value, grad_h_value = compute_and_eval_h_and_grad_batch(
        robot_state,
        np.asarray(obstacle_state, dtype=float)[None, :],
        robot_radius,
        obstacle_radius,
        n,
        tau,
    )
    return float(b[0]), float(a[0]), float(y_star[0]), float(h_value[0]), grad_h_value[0]
