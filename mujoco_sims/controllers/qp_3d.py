from .sh_cbf_core import class_K_function
from .sh_cbf_jax import compute_candidate_h
import scipy.sparse as sparse
import osqp
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

# h and its gradient w.r.t. the robot state, by automatic differentiation
_candidate_h_value_and_grad = jax.jit(
    jax.value_and_grad(compute_candidate_h),
    static_argnames=("n",),
)


class QP3D:
    """
    Solves the simple acceleration ref tracking qp 

    State:
        x = [x, y, z, vx, vy, vz]

    Input:
        u = [ax, ay, az]
    """

    def __init__(
        self,
        dt,
        target=np.array([3.0, 3.0, 3.0]),
        initial_state=np.zeros(6),
        sh_n=6,
        sh_tau=1.2,
        collision_radius=0.5,
        obstacles=[],
        slowing_distance=0.2,
        velocity_time_constant=0.5,
    ):
        self.step = 0
        self.target = target
        self.dt = dt
        self.state = np.asarray(initial_state, dtype=float)

        self.sh_n = sh_n
        self.sh_tau = sh_tau
        self.collision_radius = collision_radius

        # Preferred velocity: reduced within slowing_distance of the target and
        # tracked with time constant velocity_time_constant
        self.slowing_distance = slowing_distance
        self.velocity_time_constant = velocity_time_constant

        # commanded [ax, ay, az]
        self.cmd_accel = np.zeros(3)

        # Reference speed to track
        self.reference_speed = 0.2

        # Double integrator input matrix
        self.G = np.array([[
            0, 0, 0], [
            0, 0, 0], [
            0, 0, 0], [
            1, 0, 0], [
            0, 1, 0], [
            0, 0, 1]],
            dtype=float)


    # ==============================================================
    # UPDATE DATA
    # ==============================================================

    def update_state(self, state):
        self.state = state

    def set_target(self, target):
        self.target = target

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_reference_speed(self, reference_speed):
        self.reference_speed = reference_speed

    def set_max_accel(self, max_accel):
        self.max_accel = max_accel

    def increment_step(self):
        self.step = self.step + 1

    def compute_acceleration_reference(self) -> np.ndarray:
        """
        Tracks the preferred velocity of the VO paradigm: the direction to the
        target scaled by the reference speed, reduced linearly within
        slowing_distance of the target so that the agent stops on it.
        """
        position = self.state[:3]
        velocity = self.state[3:]

        error = self.target - position

        preferred_velocity = (
            self.reference_speed
            * error
            / max(np.linalg.norm(error), self.slowing_distance)
        )

        return (preferred_velocity - velocity) / self.velocity_time_constant

    # ==============================================================
    # COMPUTE OPTIMAL COMMAND
    # ==============================================================

    def compute_command(self):

        acc_ref = self.compute_acceleration_reference()

        # Constraints
        active_constraints = 0

        acceleration_lower_bound = (
            -self.max_accel * np.ones(3, dtype=float)
        )

        acceleration_upper_bound = (
            self.max_accel * np.ones(3, dtype=float)
        )

        constraint_rows = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        ]

        constraint_lower_bounds = [
            acceleration_lower_bound[0],
            acceleration_lower_bound[1],
            acceleration_lower_bound[2],
        ]

        constraint_upper_bounds = [
            acceleration_upper_bound[0],
            acceleration_upper_bound[1],
            acceleration_upper_bound[2],
        ]

        constraint_start_time = time.time()

        # -------------------------------------------------
        # CBF constraints - 3D SH
        # -------------------------------------------------
        # Recall: obstacle structure
        #   { obstacle_name: {collision_radius: 0.1, p: [px, py,pz], v:[vx, vy, vz]}}
        for obstacle_name, obstacle in self.obstacles.items():
            obstacle_distance = np.linalg.norm(self.state[:3] - obstacle['p'])

            # The constraint is kept when overlapping: the CBF then only
            # forbids approaching the obstacle (see compute_candidate_h)
            if obstacle_distance <= (self.collision_radius + obstacle['collision_radius']):
                print(
                    f"Overlapping {obstacle_name}: "
                    f"distance={obstacle_distance:.3f}"
                )

            cbf_obstacle_state = np.asarray(
                np.concatenate((obstacle['p'], obstacle['v'])),
                dtype=float,
            )

            h_value, grad_h = _candidate_h_value_and_grad(
                self.state,
                cbf_obstacle_state,
                self.collision_radius,
                obstacle['collision_radius'],
                n=self.sh_n,
                tau=self.sh_tau,
            )

            h_value = float(h_value)
            grad_h = np.asarray(grad_h, dtype=float)

            class_k = class_K_function(h_value, gamma=100.0, beta=0)

            # -------------------------------------------------
            # CBF constraint, with h depending on p_B - p_A and the
            # obstacle moving at constant velocity:
            #
            # L_f h = grad_h[:3] (v_A - v_B),  L_g h = grad_h G
            #
            # L_g h u >= -(L_f h + class_k)
            #
            # OSQP representation:
            #
            # lower_i <= control_row @ u <= +inf
            # -------------------------------------------------
            control_row = np.asarray(grad_h @ self.G, dtype=float).reshape(3)

            relative_velocity = self.state[3:] - obstacle['v']

            drift_and_class_k = float(grad_h[:3] @ relative_velocity + class_k)

            cbf_lower_bound = -drift_and_class_k

            constraint_rows.append(control_row)
            constraint_lower_bounds.append(cbf_lower_bound)
            constraint_upper_bounds.append(np.inf)

            active_constraints += 1

            cbf_at_reference = (drift_and_class_k + control_row @ acc_ref)

            # if self.step % 80 == 0:
            #     print(
            #         f"ob {obstacle_name} "
            #         f"(pos={obstacle['p']}, vel={obstacle['v']}) "
            #         f"h={float(h_value):.6f}, "
            #         f"CBF(u_ref)={cbf_at_reference:.6f} >= 0, "
            #         f"row={control_row}, "
            #         f"lower={cbf_lower_bound:.6f}"
            #     )

        # print(f"Building constraints took {time.time() - constraint_start_time: 6.3f}s")

        lower = np.asarray(
            constraint_lower_bounds,
            dtype=float,
        )

        upper = np.asarray(
            constraint_upper_bounds,
            dtype=float,
        )

        # Solver setup
        solver = osqp.OSQP()

        P = sparse.eye(
            3,
            format="csc",
            dtype=float,
        )

        q = -acc_ref

        A_qp = sparse.csc_matrix(
            np.vstack(constraint_rows),
            dtype=float,
        )

        solver.setup(
            P=P,
            q=q,
            A=A_qp,
            l=lower,
            u=upper,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=100
        )

        results = solver.solve()
        status = results.info.status.lower()

        if (
            results.x is not None
            and status.startswith("solved")
        ):
            u_star = np.asarray(
                results.x,
                dtype=float,
            )

            self.previous_solution = u_star.copy()
            self.cmd_accel = u_star.copy()
        else:
            self.cmd_accel = np.zeros(3)

        if self.step % 80 == 0:
            print(
                f"OSQP status={results.info.status}, "
                f"iterations={results.info.iter}, "
                f"primal_residual={results.info.prim_res:.3e}, "
                f"dual_residual={results.info.dual_res:.3e}, "
                f"active_constraints={active_constraints}, "
                f"acc_ref={' '.join(f'{a:3.6f}' for a in acc_ref)}, "
                f"acc_cmd={' '.join(f'{a:3.6f}' for a in self.cmd_accel)}"
            )

        return self.cmd_accel
