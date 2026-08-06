from .sh_cbf_core import compute_and_eval_h_and_grad, class_K_function
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.interpolate import BPoly
import osqp
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)



def hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)

    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def vee(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)

    return np.array([
        matrix[2, 1],
        matrix[0, 2],
        matrix[1, 0],
    ])


def gain_matrix(gain: float | np.ndarray) -> np.ndarray:
    gain = np.asarray(gain, dtype=float)

    if gain.ndim == 0:
        return float(gain) * np.eye(3)

    if gain.shape == (3,):
        return np.diag(gain)

    if gain.shape == (3, 3):
        return gain

    raise ValueError(
        "Gain must be a scalar, length-3 vector, or 3x3 matrix."
    )
    
    
def normalize(
    vector: np.ndarray,
    fallback: np.ndarray,
    tolerance: float = 1e-9,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm < tolerance:
        return np.asarray(fallback, dtype=float).copy()

    return vector / norm


class QP3DPrecompDrone:
    """
    Geometric position controller combined with an actual-input CBF-QP.

    State:
        x = [
            position_world,       # 3
            velocity_world,       # 3
            R_body_to_world,      # 9, row-major
            angular_velocity_body # 3
        ]

    Input:
        u = [
            total_thrust,
            body_moment_x,
            body_moment_y,
            body_moment_z
        ]

    Convention:
        - World +z points upward.
        - Gravity acceleration is -g * e3.
        - Positive thrust acts along +R @ e3.
    """

    

    def __init__(
        self,
        dt,
        mass,
        inertia,
        target=np.array([0.0, 0.0, 0.0]),
        initial_state: np.ndarray | None = None,
        position_kp: float | np.ndarray = 4.0,
        position_kd: float | np.ndarray = 3.5,
        attitude_kp: float | np.ndarray = 6.0,
        attitude_kd: float | np.ndarray = 1.5,
        min_thrust: float = 0.0,
        max_thrust: float = 30.0,
        max_moment: float | np.ndarray = 2.0,
        gravity: float = 9.81,
        sh_n: int = 6,
        sh_tau: float = 1.2,
        collision_radius: float = 0.5,
        obstacles: dict | None = None,
        radius_tolerance: float = 0.01,
        cbf_gamma: float = 100.0,
    ):
        self.dt = float(dt)
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.J = np.asarray(inertia, dtype=float)

        if self.J.shape != (3, 3):
            raise ValueError("inertia must have shape (3, 3)")

        self.J_inverse = np.linalg.inv(self.J)

        self.e3 = np.array([0.0, 0.0, 1.0])

        self.target = (
            np.zeros(3)
            if target is None
            else np.asarray(target, dtype=float)
        )

        if initial_state is None:
            self.state = np.concatenate([
                np.zeros(3),             # position
                np.zeros(3),             # velocity
                np.eye(3).reshape(-1),   # rotation
                np.zeros(3),             # angular velocity
            ])
        else:
            self.state = np.asarray(initial_state, dtype=float)

        if self.state.shape != (18,):
            raise ValueError("initial_state must have shape (18,): [p, v, R.reshape(9), omega]")

        self.Kx = gain_matrix(position_kp)
        self.Kv = gain_matrix(position_kd)
        self.KR = gain_matrix(attitude_kp)
        self.KOmega = gain_matrix(attitude_kd)

        self.min_thrust = float(min_thrust)
        self.max_thrust = float(max_thrust)
        self.max_moment = np.broadcast_to(np.asarray(max_moment, dtype=float),(3,),).copy()
        
        self.sh_n = int(sh_n)
        self.sh_tau = float(sh_tau)
        self.collision_radius = float(collision_radius)
        self.radius_tolerance = float(radius_tolerance)
        self.cbf_gamma = float(cbf_gamma)

        self.obstacles = {} if obstacles is None else obstacles

        self.step = 0

        self.previous_desired_rotation = None
        self.previous_desired_angular_velocity = None

        # [thrust, Mx, My, Mz]
        self.command = np.array([
            self.mass * self.gravity,
            0.0,
            0.0,
            0.0,
        ])

        # Larger values make the QP less willing to change that input.
        self.input_weights = np.diag([
            1.0,
            1.0,
            1.0,
            1.0,
        ])


    # ==============================================================
    # UPDATE DATA
    # ==============================================================

    def unpack_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position = self.state[0:3]
        velocity = self.state[3:6]
        rotation = self.state[6:15].reshape(3, 3)
        angular_velocity = self.state[15:18]

        return position, velocity, rotation, angular_velocity

    def update_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float)

        if state.shape != (18,):
            raise ValueError("state must have shape (18,): [p, v, R.reshape(9), omega]")

        self.state = state

    def set_target(
        self,
        target: np.ndarray,
    ) -> None:
        self.target = np.asarray(
            target,
            dtype=float,
        )

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles

    def increment_step(self):
        self.step = self.step + 1
        
         # ==============================================================
    # Nonlinear control-affine drone model
    # ==============================================================

    def nonlinear_dynamics(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns drift f(x) and input matrix g(x) such that:

            x_dot = f(x) + g(x) @ u

        where:

            u = [thrust, Mx, My, Mz]
        """
        _, velocity, rotation, angular_velocity = (
            self.unpack_state()
        )

        drift = np.zeros(18, dtype=float)
        input_matrix = np.zeros((18, 4), dtype=float)

        # p_dot = v
        drift[0:3] = velocity
        # v_dot = -g*e3 + thrust/m * R*e3
        drift[3:6] = -self.gravity * self.e3
        input_matrix[3:6, 0] = (rotation @ self.e3) / self.mass
        # R_dot = R * hat(omega)
        rotation_dot = rotation @ hat(angular_velocity)
        drift[6:15] = rotation_dot.reshape(-1)
        # omega_dot = J^-1(M - omega x J*omega)
        drift[15:18] = -self.J_inverse @ np.cross(angular_velocity, self.J @ angular_velocity,)

        input_matrix[15:18, 1:4] = self.J_inverse

        return drift, input_matrix
    
     # ==============================================================
    # Desired geometric reference
    # ==============================================================

    def construct_desired_rotation(
        self,
        desired_force: np.ndarray,
        current_rotation: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the desired attitude using only the force direction.

        The unconstrained yaw degree of freedom is selected by preserving
        the drone's current horizontal heading.
        """
        desired_force = np.asarray(desired_force, dtype=float)
        R = np.asarray(current_rotation, dtype=float)

        force_norm = np.linalg.norm(desired_force)

        if force_norm < 1e-9:
            desired_b3 = np.array([0.0, 0.0, 1.0])
        else:
            desired_b3 = desired_force / force_norm

        # Current body x-axis expressed in the world frame.
        current_b1 = R[:, 0]

        # Preserve only its horizontal heading.
        heading_direction = np.array([
            current_b1[0],
            current_b1[1],
            0.0,
        ])

        heading_norm = np.linalg.norm(heading_direction)

        if heading_norm < 1e-8:
            heading_direction = np.array([1.0, 0.0, 0.0])
        else:
            heading_direction /= heading_norm

        desired_b2 = np.cross(
            desired_b3,
            heading_direction,
        )

        # Handle the singular case where the force axis is parallel
        # to the selected heading direction.
        if np.linalg.norm(desired_b2) < 1e-8:
            alternatives = np.eye(3)

            best_axis = alternatives[
                np.argmin(np.abs(alternatives @ desired_b3))
            ]

            desired_b2 = np.cross(
                desired_b3,
                best_axis,
            )

        desired_b2 /= np.linalg.norm(desired_b2)

        desired_b1 = np.cross(
            desired_b2,
            desired_b3,
        )

        desired_b1 /= np.linalg.norm(desired_b1)

        return np.column_stack((
            desired_b1,
            desired_b2,
            desired_b3,
        ))
    
    def desired_attitude_derivatives(
        self,
        desired_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Numerical approximation of desired angular velocity and angular acceleration.
        For aggressive trajectories, analytical jerk/snap-based derivatives are preferable.
        """
        if self.previous_desired_rotation is None:
            desired_angular_velocity = np.zeros(3)
            desired_angular_acceleration = np.zeros(3)

        else:
            desired_rotation_dot = (
                desired_rotation
                - self.previous_desired_rotation
            ) / self.dt

            desired_omega_hat = (
                desired_rotation.T
                @ desired_rotation_dot
            )

            # Project numerical result onto so(3).
            desired_omega_hat = 0.5 * (
                desired_omega_hat
                - desired_omega_hat.T
            )

            desired_angular_velocity = vee(
                desired_omega_hat
            )

            if self.previous_desired_angular_velocity is None:
                desired_angular_acceleration = np.zeros(3)
            else:
                desired_angular_acceleration = (
                    desired_angular_velocity
                    - self.previous_desired_angular_velocity
                ) / self.dt

        self.previous_desired_rotation = (
            desired_rotation.copy()
        )

        self.previous_desired_angular_velocity = (
            desired_angular_velocity.copy()
        )

        return (
            desired_angular_velocity,
            desired_angular_acceleration,
        )

    def compute_geometric_reference(self):
        position, velocity, rotation, angular_velocity = (
            self.unpack_state()
        )

        position_error = (
            position - self.target
        )

        # Static position target:
        # desired velocity     = 0
        # desired acceleration = 0
        velocity_error = velocity

        desired_force = (
            -self.Kx @ position_error
            -self.Kv @ velocity_error
            +self.mass * self.gravity * self.e3
        )

        desired_rotation = self.construct_desired_rotation(
            desired_force=desired_force,
            current_rotation=rotation,
        )

        (
            desired_angular_velocity,
            desired_angular_acceleration,
        ) = self.desired_attitude_derivatives(
            desired_rotation
        )

        attitude_error = 0.5 * vee(
            desired_rotation.T @ rotation
            - rotation.T @ desired_rotation
        )

        angular_velocity_error = (
            angular_velocity
            - rotation.T
            @ desired_rotation
            @ desired_angular_velocity
        )

        # Thrust acts only along the current body z-axis.
        thrust_reference = float(
            desired_force @ (rotation @ self.e3)
        )

        moment_reference = (
            -self.KR @ attitude_error
            -self.KOmega @ angular_velocity_error
            +np.cross(
                angular_velocity,
                self.J @ angular_velocity,
            )
            -self.J
            @ (
                hat(angular_velocity)
                @ rotation.T
                @ desired_rotation
                @ desired_angular_velocity
                -rotation.T
                @ desired_rotation
                @ desired_angular_acceleration
            )
        )

        control_reference = np.concatenate((
            [thrust_reference],
            moment_reference,
        ))

        diagnostics = {
            "position_error": position_error,
            "velocity_error": velocity_error,
            "desired_force": desired_force,
            "desired_rotation": desired_rotation,
            "attitude_error": attitude_error,
            "angular_velocity_error": angular_velocity_error,
            "thrust_reference": thrust_reference,
            "moment_reference": moment_reference,
        }

        return control_reference, diagnostics


    # ==============================================================
    # COMPUTE OPTIMAL COMMAND
    # ==============================================================

    def compute_command(self):
        control_reference, controller_data = (
            self.compute_geometric_reference()
        )

        position, velocity, rotation, _ = (
            self.unpack_state()
        )

        # Actual input:
        # [thrust, Mx, My, Mz]
        input_lower_bound = np.concatenate([
            [self.min_thrust],
            -self.max_moment,
        ])

        input_upper_bound = np.concatenate([
            [self.max_thrust],
            self.max_moment,
        ])

        # Correct identity rows for the four actuator bounds.
        constraint_rows = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ]

        constraint_lower_bounds = list(input_lower_bound)
        constraint_upper_bounds = list(input_upper_bound)

        obstacles_boundaries = {}
        active_constraints = 0

        thrust_axis_world = rotation @ self.e3

        # The SH-CBF still evaluates the translational state.
        translational_state = np.concatenate([
            position,
            velocity,
        ])

        for obstacle_name, obstacle in self.obstacles.items():
            obstacle_position = np.asarray(
                obstacle["p"],
                dtype=float,
            )

            obstacle_velocity = np.asarray(
                obstacle.get("v", np.zeros(3)),
                dtype=float,
            )

            obstacle_acceleration = np.asarray(
                obstacle.get("a", np.zeros(3)),
                dtype=float,
            )

            obstacle_state = np.concatenate([
                obstacle_position,
                obstacle_velocity,
            ])

            (
                boundary_b,
                boundary_a,
                vy_tan,
                h_value,
                grad_h_value,
            ) = compute_and_eval_h_and_grad(
                robot_state=translational_state,
                obstacle_state=obstacle_state,
                robot_radius=(self.collision_radius + self.radius_tolerance),
                obstacle_radius=obstacle["collision_radius"],
                n=self.sh_n,
                tau=self.sh_tau,
            )

            gradient = np.asarray(
                grad_h_value,
                dtype=float,
            ).reshape(6)

            gradient_position = gradient[0:3]
            gradient_velocity = gradient[3:6]

            # Relative translational drift:
            #
            # p_rel_dot = v_robot - v_obstacle
            #
            # v_rel_dot =
            #     -g*e3
            #     + thrust/m * R*e3
            #     - a_obstacle
            #
            relative_position_drift = (
                velocity - obstacle_velocity
            )

            relative_velocity_drift = (
                -self.gravity * self.e3
                -obstacle_acceleration
            )

            lie_f_h = float(
                gradient_position
                @ relative_position_drift
                +gradient_velocity
                @ relative_velocity_drift
            )

            # The SH-CBF only has relative degree one with respect
            # to total thrust.
            #
            # Moments do not appear in h_dot.
            lie_g_h = np.array([
                float(
                    gradient_velocity
                    @ thrust_axis_world
                    / self.mass
                ),
                0.0,
                0.0,
                0.0,
            ])

            class_k = float(
                class_K_function(
                    h_value,
                    gamma=self.cbf_gamma,
                    beta=0,
                )
            )

            # Lf h + Lg h u + alpha(h) >= 0
            #
            # Lg h u >= -(Lf h + alpha(h))
            cbf_lower_bound = -(lie_f_h + class_k)

            constraint_rows.append(lie_g_h)
            constraint_lower_bounds.append(
                cbf_lower_bound
            )
            constraint_upper_bounds.append(np.inf)

            cbf_at_reference = float(
                lie_f_h
                + lie_g_h @ control_reference
                + class_k
            )

            obstacle_distance = np.linalg.norm(
                position - obstacle_position
            )

            obstacles_boundaries[obstacle_name] = {
                "a": boundary_a,
                "b": boundary_b,
                "vy_tan": vy_tan,
                "h_value": float(h_value),
                "gradient": gradient,
                "lie_f_h": lie_f_h,
                "lie_g_h": lie_g_h,
                "class_k": class_k,
                "cbf_lower_bound": cbf_lower_bound,
                "cbf_at_reference": cbf_at_reference,
                "distance": obstacle_distance,
            }

            active_constraints += 1

            if (
                abs(lie_g_h[0]) < 1e-8
                and cbf_lower_bound > 0.0
            ):
                print(
                    f"Warning: {obstacle_name} CBF cannot be "
                    "affected by thrust in the current attitude. "
                    f"Lg_h={lie_g_h[0]:.3e}, "
                    f"required={cbf_lower_bound:.3e}"
                )

        lower = np.asarray(
            constraint_lower_bounds,
            dtype=float,
        )

        upper = np.asarray(
            constraint_upper_bounds,
            dtype=float,
        )

        A_qp = sparse.csc_matrix(
            np.vstack(constraint_rows),
            dtype=float,
        )

        P = sparse.csc_matrix(
            self.input_weights,
            dtype=float,
        )

        q = -self.input_weights @ control_reference

        solver = osqp.OSQP()

        solver.setup(
            P=P,
            q=q,
            A=A_qp,
            l=lower,
            u=upper,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=200,
            polish=True,
        )

        result = solver.solve()
        status = result.info.status.lower()

        if (
            result.x is not None
            and status.startswith("solved")
        ):
            self.command = np.asarray(
                result.x,
                dtype=float,
            )

        else:
            # This fallback is not guaranteed safe.
            self.command = np.clip(
                control_reference,
                input_lower_bound,
                input_upper_bound,
            )

            print(
                f"CBF-QP failed: {result.info.status}. "
                "Using clipped geometric reference; "
                "safety is not guaranteed."
            )

        if self.step % 80 == 0:
            print(
                f"OSQP status={result.info.status}, "
                f"iterations={result.info.iter}, "
                f"active_constraints={active_constraints}\n"
                f"u_ref={control_reference}\n"
                f"u_cmd={self.command}"
            )

        return (
            self.command,
            obstacles_boundaries,
            controller_data,
        )
