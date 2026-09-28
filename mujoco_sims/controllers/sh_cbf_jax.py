from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def compute_sh_parameters(ell, R, n, tau, num_iters=64):
    """
    JAX counterpart of sh_cbf_core.compute_sh_parameters, returning a and b.

    The tangency ordinate y_star is found with a fixed number of bisection
    steps and is excluded from differentiation: b is stationary in y at y_star,
    so only the explicit dependence of b on ell remains in its gradient.
    """
    d = ell / tau
    r = R / tau
    a = d - r

    if n == 2:
        # The hyperbola osculates the disk at the vertex
        return a, jnp.sqrt(a * r)

    a_, d_, r_ = jax.lax.stop_gradient((a, d, r))

    # Tangency polynomial P(y) deflated by its root y = a, negative at a and
    # positive at d - r^2/d (Lemma 1)
    def deflated_tangency_poly(y):
        s = sum(y**(n - 1 - k) * a_**k for k in range(n))  # (y^n - a^n) / (y - a)
        return (d_ + r_ - y) * y**(n - 1) + (y - d_) * s

    def bisection_step(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        below = deflated_tangency_poly(mid) < 0.0
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, num_iters, bisection_step, (a_, d_ - r_**2 / d_))
    y_star = 0.5 * (lo + hi)

    b = a * jnp.sqrt(r**2 - (y_star - d)**2) / (y_star**n - a**n)**(1.0 / n)

    return a, b


@partial(jax.jit, static_argnames=("n",))
def compute_candidate_h(
    robot_state: jnp.ndarray,
    obstacle_state: jnp.ndarray,
    robot_radius: float,
    obstacle_radius: float,
    n: int,
    tau: float,
):
    """
    JAX counterpart of the CBF of sh_cbf_core.compute_and_eval_h_and_grad, to
    be differentiated w.r.t. the robot state [p_A, v_A], in 2D or 3D.
    """
    if n < 2 or n % 2:
        raise ValueError(f"n must be an even integer >= 2, got {n}")

    eps = 1e-12
    R = robot_radius + obstacle_radius
    k = robot_state.shape[0] // 2

    # Robot -> obstacle direction
    delta_p = obstacle_state[:k] - robot_state[:k]
    ell = jnp.sqrt(delta_p @ delta_p + eps**2)
    e_los = delta_p / ell

    v_rel = robot_state[k:] - obstacle_state[k:]

    # Relative velocity along and perpendicular to the line of sight
    v_parallel = v_rel @ e_los
    v_perpendicular = v_rel - v_parallel * e_los
    v_tangential_sq = v_perpendicular @ v_perpendicular

    # Overlapping: the collision set tends to the half-space v_parallel > 0.
    # The super-hyperbola is still evaluated at a valid distance, otherwise the
    # branch discarded by jnp.where would turn the gradient into NaN.
    overlapping = ell <= R
    a, b = compute_sh_parameters(jnp.where(overlapping, 2.0 * R, ell), R, n, tau)

    # n is even, so v_t^n = (v_t^2)^(n/2) and h is smooth at v_t = 0
    phi = a * (1.0 + (v_tangential_sq / b**2) ** (n // 2)) ** (1.0 / n)

    return jnp.where(overlapping, 0.0, phi) - v_parallel
