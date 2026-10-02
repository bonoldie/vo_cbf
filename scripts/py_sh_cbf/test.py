import numpy as np
from sh_cbf_core import class_K_function, compute_and_eval_h_and_grad
import matplotlib.pyplot as plt


def main():
    # State: [x, y, vx, vy]
    robot_state = np.array([0.5, 0.0, 0.0, 0.0])
    obstacle_state = np.array([2.0, 2.0, 0.0, 0.0])

    robot_radius = 0.2
    obstacle_radius = 0.2

    tau = 1.5
    n = 6

    b, a, _, h, grad_h = compute_and_eval_h_and_grad(
        robot_state,
        obstacle_state,
        robot_radius,
        obstacle_radius,
        n,
        tau,
    )

    print(
        f"Found parameters: a={a}, b={b}"
    )

    # h depends on p_obs - p and v - v_obs, so with the obstacle moving at
    # constant velocity: L_f h = grad_h[:2] (v - v_obs), L_g h = grad_h[2:]
    lf_h = grad_h[:2] @ (robot_state[2:] - obstacle_state[2:])
    lg_h = grad_h[2:]

    # plant EQ
    u = np.array((10.0, 0.0))

    h_val = class_K_function(h, gamma=1.0, beta=0)
    U_cbf = lf_h + lg_h @ u + h_val

    print(f"gradH(robot_state): {grad_h}")
    print(f"h(robot_state): {h}")
    print(f"U_cbf: {U_cbf}")

    #### PLOTSS

    # Grid of candidate inputs in world frame
    ux_min, ux_max = -50.0, 50.0
    uy_min, uy_max = -50.0, 50.0
    num_samples = 81

    ux_vals = np.linspace(ux_min, ux_max, num_samples)
    uy_vals = np.linspace(uy_min, uy_max, num_samples)

    UX, UY = np.meshgrid(ux_vals, uy_vals)

    # Flatten grid into shape (N, 2)
    U_world_grid = np.stack(
        [UX.ravel(), UY.ravel()],
        axis=1,
    )

    U_cbf_vals = lf_h + U_world_grid @ lg_h + h_val

    safe_mask = U_cbf_vals >= 0.0

    plt.figure(figsize=(7, 7))

    plt.scatter(
        U_world_grid[~safe_mask, 0],
        U_world_grid[~safe_mask, 1],
        c="red",
        s=12,
        label="Unsafe: U_cbf < 0",
    )

    plt.scatter(
        U_world_grid[safe_mask, 0],
        U_world_grid[safe_mask, 1],
        c="green",
        s=12,
        label="Safe: U_cbf >= 0",
    )

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)

    plt.xlabel("u_x world")
    plt.ylabel("u_y world")
    plt.title("CBF-admissible input grid")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
