from .sh_cbf_core import compute_and_eval_h_and_grad, class_K_function
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.interpolate import BPoly
import osqp
import time
import numpy as np
from utils.utils import vee
 
class QP3DPrecompDrone:
    """
    Solves the simple acceleration ref tracking qp 

    State:
        x = [x, y, z, vx, vy, vz, R, omega]

    Input:  
        u = [f, M1, M2, M3]
    """
    
    def quadrotor_F_G(state, mass, J, g=9.81):
        """
        state: [p(3), v(3), R.reshape(-1)(9), omega(3)]
        input: [thrust, Mx, My, Mz]

        R maps body coordinates into world coordinates.
        omega, J and M use the same body frame.
        Gravity acts along world +z.
        """
        state = np.asarray(state, dtype=float)
        J = np.asarray(J, dtype=float)

        v = state[3:6]
        R = state[6:15].reshape(3, 3)
        omega = state[15:18]

        F = np.zeros(18)
        F[0:3] = v
        F[3:6] = [0.0, 0.0, g]
        F[6:15] = (R @ vee(omega)).reshape(-1)
        F[15:18] = -np.linalg.solve(
            J, np.cross(omega, J @ omega)
        )

        G = np.zeros((18, 4))
        G[3:6, 0] = -R[:, 2] / mass
        G[15:18, 1:4] = np.linalg.solve(J, np.eye(3))

        return F, G


    def setup_plot(self):
        plt.ion()

        self.traj_fig, self.traj_axes = plt.subplots(
            3,
            1,
            figsize=(9, 8),
            sharex=True,
        )

        labels = ["x", "y", "z"]

        self.position_lines = [
            self.traj_axes[0].plot([], [], label=label)[0]
            for label in labels
        ]

        self.velocity_lines = [
            self.traj_axes[1].plot([], [], label=label)[0]
            for label in labels
        ]

        self.acceleration_lines = [
            self.traj_axes[2].plot([], [], label=label)[0]
            for label in labels
        ]

        self.traj_axes[0].set_ylabel("Position [m]")
        self.traj_axes[1].set_ylabel("Velocity [m/s]")
        self.traj_axes[2].set_ylabel("Acceleration [m/s²]")
        self.traj_axes[2].set_xlabel("Time [s]")

        for axis in self.traj_axes:
            axis.grid(True)
            axis.legend()

        self.traj_fig.tight_layout()

    def __init__(
        self,
        dt,
        target=np.array([3.0, 3.0, 3.0]),
        initial_state=np.zeros(6),
        sh_n=6,
        sh_tau=1.2,
        collision_radius=0.5,
        obstacles=[],
        radius_tollerance = 0.01,
        mass = 1.0
    ):
        self.step = 0
        self.target = target
        self.dt = dt
        self.state = np.asarray(initial_state, dtype=float)

        self.sh_n = sh_n
        self.sh_tau = sh_tau
        self.collision_radius = collision_radius
        self.radius_tollerance = radius_tollerance
        self.mass = mass
        
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

        # self.setup_plot()

    def plot_trajectory(
        self,
        trajectory: BPoly,
        T: float,
    ) -> None:
        times = np.linspace(0.0, T, 200)

        positions = np.asarray(
            trajectory(times, nu=0),
            dtype=float,
        )

        velocities = np.asarray(
            trajectory(times, nu=1),
            dtype=float,
        )

        accelerations = np.asarray(
            trajectory(times, nu=2),
            dtype=float,
        )

        for component in range(3):
            self.position_lines[component].set_data(
                times,
                positions[:, component],
            )

            self.velocity_lines[component].set_data(
                times,
                velocities[:, component],
            )

            self.acceleration_lines[component].set_data(
                times,
                accelerations[:, component],
            )

        for axis in self.traj_axes:
            axis.set_xlim(0.0, T)
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

        self.traj_fig.canvas.draw_idle()
        self.traj_fig.canvas.flush_events()

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

    # Accelearation reference
    def compute_acceleration_reference(self) -> np.ndarray:
        position = self.state[:3]
        velocity = self.state[3:]

        error = self.target - position
        distance = np.linalg.norm(error)
        speed = np.linalg.norm(velocity)

        # Avoid numerical movement
        if distance < 0.001 and speed < 0.001:
            return np.zeros(3)

        T = max(
            0.5,
            distance / self.reference_speed
        )

        trajectory = BPoly.from_derivatives(
            [0.0, T],
            [
                [
                    position,
                    velocity,
                    self.cmd_accel,
                ],
                [
                    self.target,
                    np.zeros(3),
                    np.zeros(3),
                ],
            ],
        )

        # self.plot_trajectory(trajectory, T)

        acc_ref = np.asarray(
            trajectory(self.dt, nu=2),
            dtype=float,
        )

        # acc_norm = np.linalg.norm(acc_ref)

        # if acc_norm > self.max_accel:
        #     acc_ref *= self.max_accel / acc_norm

        return acc_ref

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

        obstacles_boundaries = {}

        # -------------------------------------------------
        # CBF constraints - 3D SH
        # -------------------------------------------------
        # Recall: obstacle structure
        #   { obstacle_name: {collision_radius: 0.1, p: [px, py,pz], v:[vx, vy, vz]}}
        for obstacle_name, obstacle in self.obstacles.items():
            obstacle_distance = np.linalg.norm(self.state[:3] - obstacle['p'])

            # The constraint is kept when overlapping: the CBF then only
            # forbids approaching the obstacle (see compute_and_eval_h_and_grad)
            if obstacle_distance <= (self.collision_radius + obstacle['collision_radius']):
                print(
                    f"Overlapping {obstacle_name}: "
                    f"distance={obstacle_distance:.3f}"
                )

            cbf_obstacle_state = np.asarray(
                np.concatenate((obstacle['p'], obstacle['v'])),
                dtype=float,
            )

            b, a, vy_tan, h_value, grad_h_value = compute_and_eval_h_and_grad(
                robot_state=self.state,
                obstacle_state=cbf_obstacle_state,
                robot_radius=self.collision_radius + self.radius_tollerance,
                obstacle_radius=obstacle['collision_radius'],
                n=self.sh_n,
                tau=self.sh_tau
            )

            obstacles_boundaries[obstacle_name] = {
                "a": a,
                "b": b,
                "h_value": h_value, 
                "grad_h_value": grad_h_value
            }

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
            control_row = np.asarray(grad_h_value @ self.G, dtype=float).reshape(3)

            relative_velocity = self.state[3:] - obstacle['v']

            drift_and_class_k = float(grad_h_value[:3] @ relative_velocity + class_k)

            cbf_lower_bound = -drift_and_class_k

            constraint_rows.append(control_row)
            constraint_lower_bounds.append(cbf_lower_bound)
            constraint_upper_bounds.append(np.inf)

            active_constraints += 1

            # cbf_at_reference = (drift_and_class_k + control_row @ acc_ref)
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

        return self.cmd_accel, obstacles_boundaries
