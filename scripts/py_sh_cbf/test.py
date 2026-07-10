import numpy as np
from sh_cbf_core import *
import matplotlib.pyplot as plt


@jax.jit
def class_K_function(h, gamma=1.0 , beta = 1.0):
    return gamma * h + beta * jnp.power(h,3)


def main(): 
    # State: [x, y, vx, vy]
    robot_state = jnp.array([0.5, 0.0, 0.0, 0.0])
    obstacle_state = jnp.array([2.0, 0.0, 0.0, 0.0])

    robot_radius = 0.2
    obstacle_radius = 0.2

    tau = 1.5
    n = 6

    a = compute_sh_a(robot_state, obstacle_state, robot_radius, obstacle_radius, tau)
    b = compute_sh_b(robot_state, obstacle_state, robot_radius, obstacle_radius, n, tau)

    print(
        f"Found parameters: a={a}, b={b}"
    )

    def h_as_function_of_robot_state(x):
        return compute_candidate_h(
            x,
            obstacle_state,
            robot_radius,
            obstacle_radius,
            n,
            tau,
        )
            
    gradH = jax.grad(h_as_function_of_robot_state)

    # plant EQ
    u = jnp.array((10.0, 0.0))
    
    A = jnp.array(((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
    G = jnp.array(((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    print(f"A: {A}")
    print(f"G: {G}")
    print(f"gradH(robot_state): {gradH(robot_state)}")

    # !Warning! 
    # Inputs must be in the same rotate frame of the relative velocities (y-axis torward the obstacle) 

    R_world_to_local = world_to_obstacle_aligned_frame(
        robot_state,
        obstacle_state,
    )

    u_local = u # R_world_to_local @ u

    print(f"u_local: {u_local}")
    
    h_val = class_K_function(h_as_function_of_robot_state(robot_state), gamma=1.0, beta=0)
    U_cbf = gradH(robot_state) @ A @ robot_state +  gradH(robot_state) @ G @ u_local + h_val

    print(f"U_cbf: {U_cbf}")

    #### PLOTSS
    print(f"A: {A}")
    print(f"G: {G}")

    grad_h = gradH(robot_state)

    print(f"gradH(robot_state): {grad_h}")
    print(f"h(robot_state): {h_val}")


    # Grid of candidate inputs in world frame
    ux_min, ux_max = -50.0, 50.0
    uy_min, uy_max = -50.0, 50.0
    num_samples = 81

    ux_vals = jnp.linspace(ux_min, ux_max, num_samples)
    uy_vals = jnp.linspace(uy_min, uy_max, num_samples)

    UX, UY = jnp.meshgrid(ux_vals, uy_vals)

    # Flatten grid into shape (N, 2)
    U_world_grid = jnp.stack(
        [UX.ravel(), UY.ravel()],
        axis=1,
    )

    def compute_u_cbf_for_u_world(u_world):
        # Inputs must be expressed in the same rotated frame as relative velocities
        # u_local = R_world_to_local @ u_world
        u_local = u_world
        return (
            grad_h @ A @ robot_state
            + grad_h @ G @ u_local
            + h_val
        )


    # Vectorize over all sampled inputs
    U_cbf_vals = jax.vmap(compute_u_cbf_for_u_world)(U_world_grid)

    safe_mask = U_cbf_vals >= 0.0

    # Convert to numpy for plotting
    U_world_grid_np = np.array(U_world_grid)
    U_cbf_vals_np = np.array(U_cbf_vals)
    safe_mask_np = np.array(safe_mask)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        U_world_grid_np[~safe_mask_np, 0],
        U_world_grid_np[~safe_mask_np, 1],
        c="red",
        s=12,
        label="Unsafe: U_cbf < 0",
    )

    plt.scatter(
        U_world_grid_np[safe_mask_np, 0],
        U_world_grid_np[safe_mask_np, 1],
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