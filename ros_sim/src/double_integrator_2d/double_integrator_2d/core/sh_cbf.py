import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def class_K_function(h, gamma=90.0 , beta = 220.0):
    return gamma * h + beta * jnp.power(h,3)

@jax.jit
def jax_rollout_all(u_sequence, robot_state0, mpc_dt):
    """
    Roll out double-integrator dynamics.

    u_sequence:
        [ax0, ay0, ax1, ay1, ...]

    robot_state0:
        [x, y, vx, vy]
    """

    U = u_sequence.reshape((-1, 2))

    def step(state, u):
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]

        ax = u[0]
        ay = u[1]

        vx_next = vx + ax * mpc_dt
        vy_next = vy + ay * mpc_dt

        x_next = x + vx_next * mpc_dt
        y_next = y + vy_next * mpc_dt

        next_state = jnp.array([x_next, y_next, vx_next, vy_next])

        return next_state, next_state

    _, states = jax.lax.scan(step, robot_state0, U)

    return states


@jax.jit
def jax_rollout_start_states(u_sequence, robot_state0, mpc_dt):
    """
    Returns states at the start of each MPC step.

    If N = horizon:

        start_states[0] = current state
        start_states[1] = state after u0
        start_states[2] = state after u0, u1
        ...

    Shape:
        (N, 4)
    """

    U = u_sequence.reshape((-1, 2))
    N = U.shape[0]

    def step(state, u):
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]

        ax = u[0]
        ay = u[1]

        vx_next = vx + ax * mpc_dt
        vy_next = vy + ay * mpc_dt

        x_next = x + vx_next * mpc_dt
        y_next = y + vy_next * mpc_dt

        next_state = jnp.array([x_next, y_next, vx_next, vy_next])

        return next_state, next_state

    _, states_after = jax.lax.scan(step, robot_state0, U)

    # states_after has:
    #   after u0, after u1, ..., after uN-1
    #
    # start states should be:
    #   current, after u0, after u1, ..., after uN-2
    start_states = jnp.concatenate(
        [
            robot_state0[None, :],
            states_after[:-1, :],
        ],
        axis=0,
    )

    return start_states

@partial(jax.jit, static_argnames=("n",))
def jax_sh_tangency_y(d, R, tau, n):
    """
    JAX bisection solver for SH-VO tangency y_t.

    Solves:
        d y^n - a^n y - (d^2 - R^2) y^(n-1) + d a^n = 0
    """

    eps = 1e-9

    d_safe = jnp.maximum(d, R + 1e-6)
    a = jnp.maximum((d_safe - R) / tau, eps)

    lo = a + eps
    hi = d_safe - eps

    def f(y):
        return (
            d_safe * y**n
            - a**n * y
            - (d_safe * d_safe - R * R) * y ** (n - 1)
            + d_safe * a**n
        )

    f_lo = f(lo)

    def body(_, carry):
        lo, hi, f_lo = carry

        mid = 0.5 * (lo + hi)
        f_mid = f(mid)

        use_left = f_lo * f_mid <= 0.0

        new_hi = jnp.where(use_left, mid, hi)
        new_lo = jnp.where(use_left, lo, mid)
        new_f_lo = jnp.where(use_left, f_lo, f_mid)

        return new_lo, new_hi, new_f_lo

    lo, hi, _ = jax.lax.fori_loop(
        0,
        60,
        body,
        (lo, hi, f_lo),
    )

    return 0.5 * (lo + hi)


@partial(jax.jit, static_argnames=("n",))
def jax_shvo_h(robot_state, obstacle_state, robot_radius, obstacle_radius, safety_margin, tau, n):
    """
    SH-VO barrier h.

    robot_state:
        [x_r, y_r, vx_r, vy_r]

    obstacle_state:
        [x_ob, y_ob, vx_ob, vy_ob]

    h >= 0 means the relative velocity is outside the unsafe SH-VO region.
    """

    eps = 1e-9

    x_r = robot_state[0]
    y_r = robot_state[1]
    vx_r = robot_state[2]
    vy_r = robot_state[3]

    x_ob = obstacle_state[0]
    y_ob = obstacle_state[1]
    vx_ob = obstacle_state[2]
    vy_ob = obstacle_state[3]

    dx = x_ob - x_r
    dy = y_ob - y_r

    d = jnp.sqrt(dx * dx + dy * dy + eps)

    R = robot_radius + obstacle_radius + safety_margin

    d_safe = jnp.maximum(d, R + 1e-6)

    # Local frame:
    # e_y points from robot to obstacle.
    # e_x is the lateral direction.
    e_y_x = dx / d
    e_y_y = dy / d

    e_x_x = -e_y_y
    e_x_y = e_y_x

    v_rel_x_world = vx_r - vx_ob
    v_rel_y_world = vy_r - vy_ob

    vx_rel = v_rel_x_world * e_x_x + v_rel_y_world * e_x_y
    vy_rel = v_rel_x_world * e_y_x + v_rel_y_world * e_y_y

    a = jnp.maximum((d_safe - R) / tau, eps)

    y_t = jax_sh_tangency_y(d_safe, R, tau, n)

    vx_t_sq = R * R - (y_t - d_safe) * (y_t - d_safe)
    vx_t = jnp.sqrt(jnp.maximum(vx_t_sq, eps))

    denom = jnp.maximum(y_t**n - a**n, eps)

    b = (a * vx_t) / (denom ** (1.0 / n))
    b = jnp.maximum(jnp.abs(b), eps)

    ratio = vx_rel / b

    boundary = a * (1.0 + ratio**n) ** (1.0 / n)

    h = boundary - vy_rel

    # If already geometrically overlapping, force an unsafe h.
    h = jnp.where(d > R + 1e-6, h, -1.0)

    return h

@partial(jax.jit, static_argnames=("k", "n"))
def jax_shvo_cbf_constraint_u(
    u_sequence,
    robot_state0,
    obstacle_state0,
    cbf_params,
    k,
    n,
):
    """
    NLopt-compatible SH-VO CBF constraint.

    Returns g(u) <= 0.

    CBF:
        h_dot + alpha h >= 0

    NLopt:
        g(u) = -(h_dot + alpha h) <= 0

    cbf_params:
        [
            mpc_dt,
            robot_radius,
            obstacle_radius,
            safety_margin,
            tau,
            alpha
        ]
    """

    mpc_dt = cbf_params[0]
    robot_radius = cbf_params[1]
    obstacle_radius = cbf_params[2]
    safety_margin = cbf_params[3]
    tau = cbf_params[4]
    alpha = cbf_params[5]

    start_states = jax_rollout_start_states(
        u_sequence,
        robot_state0,
        mpc_dt,
    )

    robot_k = start_states[k]

    # Constant-velocity obstacle prediction.
    t = k * mpc_dt

    obstacle_k = jnp.array(
        [
            obstacle_state0[0] + obstacle_state0[2] * t,
            obstacle_state0[1] + obstacle_state0[3] * t,
            obstacle_state0[2],
            obstacle_state0[3],
        ]
    )

    ax = u_sequence[2 * k]
    ay = u_sequence[2 * k + 1]

    h = jax_shvo_h(
        robot_k,
        obstacle_k,
        robot_radius,
        obstacle_radius,
        safety_margin,
        tau,
        n,
    )

    grad_robot = jax.grad(jax_shvo_h, argnums=0)(
        robot_k,
        obstacle_k,
        robot_radius,
        obstacle_radius,
        safety_margin,
        tau,
        n,
    )

    grad_obstacle = jax.grad(jax_shvo_h, argnums=1)(
        robot_k,
        obstacle_k,
        robot_radius,
        obstacle_radius,
        safety_margin,
        tau,
        n,
    )

    # Robot dynamics:
    # x_dot  = vx
    # y_dot  = vy
    # vx_dot = ax
    # vy_dot = ay
    f_robot = jnp.array(
        [
            robot_k[2],
            robot_k[3],
            ax,
            ay,
        ]
    )

    # Obstacle dynamics:
    # x_ob_dot  = vx_ob
    # y_ob_dot  = vy_ob
    # vx_ob_dot = 0
    # vy_ob_dot = 0
    f_obstacle = jnp.array(
        [
            obstacle_k[2],
            obstacle_k[3],
            0.0,
            0.0,
        ]
    )

    h_dot = (
        jnp.dot(grad_robot, f_robot)
        + jnp.dot(grad_obstacle, f_obstacle)
    )

    cbf_value = h_dot + alpha * h

    return -cbf_value


jax_shvo_cbf_constraint_u_grad = jax.jit(
    jax.grad(jax_shvo_cbf_constraint_u, argnums=0),
    static_argnames=("k", "n"),
)

@jax.jit
def jax_mpc_objective(u_sequence, robot_state0, target_position, objective_params):
    """
    JAX objective.

    objective_params:
        [
            mpc_dt,
            q_position,
            q_terminal,
            q_cruising_speed,
            cruising_speed,
            q_heading,
            r_acceleration
        ]
    """

    mpc_dt = objective_params[0]
    q_position = objective_params[1]
    q_terminal = objective_params[2]
    q_cruising_speed = objective_params[3]
    cruising_speed = objective_params[4]
    q_heading = objective_params[5]
    r_acceleration = objective_params[6]

    eps = 1e-9

    states = jax_rollout_all(u_sequence, robot_state0, mpc_dt)
    U = u_sequence.reshape((-1, 2))

    pos = states[:, 0:2]
    vel = states[:, 2:4]

    err = target_position[None, :] - pos

    dist_sq = jnp.sum(err * err, axis=1)
    speed = jnp.sqrt(jnp.sum(vel * vel, axis=1) + eps)

    target_dir = err / (jnp.sqrt(dist_sq + eps)[:, None])
    vel_dir = vel / speed[:, None]

    heading_alignment = jnp.sum(target_dir * vel_dir, axis=1)
    heading_cost = 1.0 - heading_alignment

    speed_error = speed - cruising_speed

    accel_sq = jnp.sum(U * U, axis=1)

    running_position_cost = q_position * jnp.sum(dist_sq)
    terminal_position_cost = q_terminal * dist_sq[-1]
    speed_cost = q_cruising_speed * jnp.sum(speed_error * speed_error)
    heading_cost_total = q_heading * jnp.sum(heading_cost)
    acceleration_cost = r_acceleration * jnp.sum(accel_sq)

    total = (
        running_position_cost
        + terminal_position_cost
        + speed_cost
        + heading_cost_total
        + acceleration_cost
    )

    return total


jax_mpc_objective_grad = jax.jit(
    jax.grad(jax_mpc_objective, argnums=0)
)