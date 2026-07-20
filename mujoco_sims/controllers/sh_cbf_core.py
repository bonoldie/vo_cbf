import jax
import jax.numpy as jnp
import numpy as np
import jaxopt 
from functools import partial

from jax import config
config.update("jax_enable_x64", True)


def world_to_obstacle_aligned_frame(robot_state, obstacle_state):
    """
    Returns R such that:

        v_local = R @ v_world

    local y-axis points from robot to obstacle.
    local x-axis is perpendicular to local y-axis.
    """

    p_robot = robot_state[:2]
    p_obstacle = obstacle_state[:2]

    # local +y axis in world coordinates: robot -> obstacle
    e_y = p_obstacle - p_robot
    e_y = e_y / (jnp.linalg.norm(e_y) + 1e-9)

    # local +x axis in world coordinates
    # This is +90 deg rotated from e_y depending on convention
    e_x = jnp.array([e_y[1], -e_y[0]])

    # Rows are local basis vectors expressed in world frame
    R_world_to_local = jnp.stack([e_x, e_y], axis=0)

    return R_world_to_local

@jax.jit
def compute_sh_a(
    robot_state: jnp.ndarray,
    obstacle_state: jnp.ndarray,
    robot_radius: float,
    obstacle_radius: float,
    tau: float,
):
    """
    Compute the a parameter of the super-hyperbola.
    """
    R = robot_radius + obstacle_radius
    d = jnp.linalg.norm(robot_state[:2] - obstacle_state[:2])

    return (d - R) / tau


@partial(jax.jit, static_argnames=("n", "num_iters"))
def get_tangency_error(b, a, n, d, r, num_iters=80):
    """
    JAX equivalent of MATLAB:

        dist_sq = @(x) x.^2 + (a * (1 + (x./b).^n).^(1/n) - d).^2;
        [~, min_dist_sq] = fminbnd(dist_sq, 0, r);
        err = sqrt(min_dist_sq) - r;

    Uses fixed-iteration golden-section search on x in [0, r].
    """

    def dist_sq(x):
        y = a * (1.0 + (x / b) ** n) ** (1.0 / n)
        return x**2 + (y - d) ** 2

    # Golden-section search constants
    golden_ratio = (1.0 + jnp.sqrt(5.0)) / 2.0
    inv_golden = 1.0 / golden_ratio

    lo = jnp.array(0.0, dtype=jnp.float64)
    hi = r

    c = hi - (hi - lo) * inv_golden
    e = lo + (hi - lo) * inv_golden

    fc = dist_sq(c)
    fe = dist_sq(e)

    def body(_, state):
        lo, hi, c, e, fc, fe = state

        choose_left = fc < fe

        new_lo = jnp.where(choose_left, lo, c)
        new_hi = jnp.where(choose_left, e, hi)

        new_c = new_hi - (new_hi - new_lo) * inv_golden
        new_e = new_lo + (new_hi - new_lo) * inv_golden

        new_fc = dist_sq(new_c)
        new_fe = dist_sq(new_e)

        return new_lo, new_hi, new_c, new_e, new_fc, new_fe

    lo, hi, c, e, fc, fe = jax.lax.fori_loop(
        0,
        num_iters,
        body,
        (lo, hi, c, e, fc, fe),
    )

    x_star = 0.5 * (lo + hi)
    min_dist_sq = dist_sq(x_star)

    return jnp.sqrt(min_dist_sq) - r


@partial(jax.jit, static_argnames=("n",))
def compute_sh_b(
    robot_state: jnp.ndarray,
    obstacle_state: jnp.ndarray,
    robot_radius: float,
    obstacle_radius: float,
    n: int,
    tau: float,
):
    """
    Compute b parameter of the super-hyperbola using Bisection (see jaxopt).
    """

    R = robot_radius + obstacle_radius
    d = jnp.linalg.norm(robot_state[:2] - obstacle_state[:2])

    a = compute_sh_a(
        robot_state,
        obstacle_state,
        robot_radius,
        obstacle_radius,
        tau,
    )

    D_term = d**2 - R**2 - a**2
    inner_term = jnp.maximum(0.0,D_term**2 - 4.0 * a**2 * R**2)

    b_2 = jnp.sqrt(0.5 * (D_term - jnp.sqrt(inner_term)))

    b_lower = jnp.array(1e-10, dtype=jnp.float64)
    b_upper = b_2 * 10.0

    def root_fun_normalized(s, a, d, R, b_lower, b_upper):
        b_guess = b_lower + s * (b_upper - b_lower)

        return get_tangency_error(
            b_guess,
            a,
            n,
            d,
            R,
        )

    # This solver select the scale factor s s.t. b =  b_lower + s_star * (b_upper - b_lower) minimize the get_tangency_error
    solver = jaxopt.Bisection(optimality_fun=root_fun_normalized, lower=0.0, upper=1.0, maxiter=80, tol=1e-10, check_bracket=False)

    out = solver.run(
        a=a,
        d=d,
        R=R,
        b_lower=b_lower,
        b_upper=b_upper,
    )


    s_star = out.params
    b_star = b_lower + s_star * (b_upper - b_lower)

    return b_star



@partial(jax.jit, static_argnames=("n",))
def compute_candidate_h(
    robot_state: jnp.ndarray,
    obstacle_state: jnp.ndarray,
    robot_radius: float,
    obstacle_radius: float,
    n: int,
    tau: float): 

    a = compute_sh_a(robot_state, obstacle_state, robot_radius, obstacle_radius, tau)
    b = compute_sh_b(robot_state, obstacle_state, robot_radius, obstacle_radius, n, tau)

    vrel = robot_state[2:] - obstacle_state[2:]

    R_world_to_local = world_to_obstacle_aligned_frame(robot_state=robot_state, obstacle_state=obstacle_state)

    vrel_local = R_world_to_local @ vrel

    return a * (1 + (vrel_local[0]/b)**n)**(1/n) - vrel_local[1]


@partial(jax.jit, static_argnames=("n",))
def compute_candidate_h_3D(
    robot_state: jnp.ndarray,
    obstacle_state: jnp.ndarray,
    robot_radius: float,
    obstacle_radius: float,
    n: int,
    tau: float): 
    # The 3D version should project the velocity on the plane defined by the robot position, the obstacle position and tangent to the relative velocity vec 

    a = compute_sh_a(robot_state, obstacle_state, robot_radius, obstacle_radius, tau)
    b = compute_sh_b(robot_state, obstacle_state, robot_radius, obstacle_radius, n, tau)

    vrel = robot_state[3:] - obstacle_state[3:]

    R_world_to_local = world_to_obstacle_aligned_frame(robot_state=robot_state, obstacle_state=obstacle_state)

    vrel_local = R_world_to_local @ vrel

    return a * (1 + (vrel_local[0]/b)**n)**(1/n) - vrel_local[1]