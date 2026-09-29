import osqp
import time
from functools import partial
from typing import Any

import jax
import numpy as np
import scipy.sparse as sparse

from .sh_cbf_jax import compute_candidate_h

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("n",))
def _candidate_h_value_and_grad(
    robot_state: jax.Array,
    obstacle_state: jax.Array,
    robot_radius: float,
    obstacle_radius: float,
    n: int,
    tau: float,
):
    """Compute h and dh/d(robot_state) in one compiled execution."""
    return jax.value_and_grad(compute_candidate_h, argnums=0)(
        robot_state,
        obstacle_state,
        robot_radius,
        obstacle_radius,
        n,
        tau,
    )


@partial(jax.jit, static_argnames=("n",))
def _batched_candidate_h_value_and_grad(
    robot_state: jax.Array,
    obstacle_states: jax.Array,
    robot_radius: float,
    obstacle_radii: jax.Array,
    n: int,
    tau: float,
):
    """Compute one h value and one robot-state gradient per obstacle."""
    value_and_grad = jax.value_and_grad(
        compute_candidate_h,
        argnums=0,
    )

    return jax.vmap(
        value_and_grad,
        in_axes=(None, 0, None, 0, None, None),
    )(
        robot_state,
        obstacle_states,
        robot_radius,
        obstacle_radii,
        n,
        tau,
    )


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _stats_ms(samples: list[float]) -> str:
    values = np.asarray(samples, dtype=float)
    return (
        f"mean={values.mean():.3f} ms, "
        f"min={values.min():.3f} ms, "
        f"p50={np.percentile(values, 50):.3f} ms, "
        f"p95={np.percentile(values, 95):.3f} ms, "
        f"max={values.max():.3f} ms"
    )


class QP3DProfiled:
    """
    Acceleration-reference tracking QP.

    State:
        x = [x, y, z, vx, vy, vz]

    Input:
        u = [ax, ay, az]

    Device selection:
        device_id >= 0: use that GPU index.
        device_id < 0:  use CPU.

    Profiling:
        profile=True prints detailed timings every profile_every steps.
        The first h+grad call normally includes JIT compilation.
    """

    def __init__(
        self,
        dt: float,
        target: np.ndarray | None = None,
        initial_state: np.ndarray | None = None,
        sh_n: int = 6,
        sh_tau: float = 1.2,
        collision_radius: float = 0.5,
        obstacles: dict[str, dict[str, Any]] | None = None,
        slowing_distance: float = 0.2,
        velocity_time_constant: float = 0.5,
        device_id: int = -1,
        profile: bool = False,
        profile_every: int = 1,
    ):
        self.step = 0
        self.target = np.asarray(
            [3.0, 3.0, 3.0] if target is None else target,
            dtype=float,
        )
        self.dt = float(dt)
        self.state = np.asarray(
            np.zeros(6) if initial_state is None else initial_state,
            dtype=float,
        )

        self.sh_n = int(sh_n)
        self.sh_tau = float(sh_tau)
        self.collision_radius = float(collision_radius)
        self.obstacles = {} if obstacles is None else obstacles

        # Preferred velocity: reduced within slowing_distance of the target and
        # tracked with time constant velocity_time_constant
        self.slowing_distance = slowing_distance
        self.velocity_time_constant = velocity_time_constant

        self.device_id = int(device_id)
        if self.device_id >= 0:
            gpu_devices = jax.devices("gpu")
            if self.device_id >= len(gpu_devices):
                raise ValueError(
                    f"Requested GPU {self.device_id}, but available GPUs are "
                    f"{gpu_devices}"
                )
            self.jax_device = gpu_devices[self.device_id]
        else:
            self.jax_device = jax.devices("cpu")[0]

        self.profile = bool(profile)
        self.profile_every = max(1, int(profile_every))
        self._h_grad_seen = False

        # Commanded [ax, ay, az].
        self.cmd_accel = np.zeros(3, dtype=float)
        self.previous_solution = np.zeros(3, dtype=float)

        self.reference_speed = 0.2
        self.max_accel = 1.0

        # This matrix is tiny and is only used after grad_h is copied to the
        # host, so keeping it as a NumPy array avoids extra JAX dispatches.
        self.G = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        print(
            f"[QP3D] JAX device={self.jax_device}; "
            f"x64={jax.config.jax_enable_x64}; profile={self.profile}"
        )

    def _should_profile(self) -> bool:
        return self.profile and self.step % self.profile_every == 0

    @staticmethod
    def _print_stage(name: str, elapsed_ms: float, indent: int = 2) -> None:
        print(f"{' ' * indent}{name:<32} {elapsed_ms:10.3f} ms")

    # ==============================================================
    # UPDATE DATA
    # ==============================================================

    def update_state(self, state):
        self.state = np.asarray(state, dtype=float)

    def set_target(self, target):
        self.target = np.asarray(target, dtype=float)

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_reference_speed(self, reference_speed):
        self.reference_speed = float(reference_speed)

    def set_max_accel(self, max_accel):
        self.max_accel = float(max_accel)

    def increment_step(self):
        self.step += 1

    # ==============================================================
    # REFERENCE
    # ==============================================================

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
            self.reference_speed * error / max(np.linalg.norm(error), self.slowing_distance)
        )

        return (preferred_velocity - velocity) / self.velocity_time_constant

    # ==============================================================
    # STANDALONE CBF BENCHMARK
    # ==============================================================

    def benchmark_cbf(
        self,
        obstacle: dict[str, Any],
        repetitions: int = 20,
    ) -> None:
        """
        Compare value-only and value+gradient steady-state costs.

        The first observed call is reported separately because it may contain
        JIT compilation. The estimated AD overhead is:

            mean(value+grad) - mean(value-only)
        """
        repetitions = max(1, int(repetitions))

        robot_device = jax.device_put(
            np.asarray(self.state, dtype=np.float64), self.jax_device
        )
        obstacle_device = jax.device_put(
            np.concatenate((obstacle["p"], obstacle["v"])).astype(np.float64),
            self.jax_device,
        )
        jax.block_until_ready((robot_device, obstacle_device))

        args = (
            robot_device,
            obstacle_device,
            self.collision_radius,
            float(obstacle["collision_radius"]),
            self.sh_n,
            self.sh_tau,
        )

        print("\n[QP3D CBF benchmark]")
        print(f"  device:      {self.jax_device}")
        print(f"  repetitions: {repetitions}")

        start_ns = time.perf_counter_ns()
        h_first = compute_candidate_h(*args)
        h_first.block_until_ready()
        first_value_ms = _elapsed_ms(start_ns)

        start_ns = time.perf_counter_ns()
        h_grad_first = _candidate_h_value_and_grad(*args)
        jax.block_until_ready(h_grad_first)
        first_value_grad_ms = _elapsed_ms(start_ns)

        value_samples: list[float] = []
        value_grad_samples: list[float] = []

        for _ in range(repetitions):
            start_ns = time.perf_counter_ns()
            h = compute_candidate_h(*args)
            h.block_until_ready()
            value_samples.append(_elapsed_ms(start_ns))

        for _ in range(repetitions):
            start_ns = time.perf_counter_ns()
            h_and_grad = _candidate_h_value_and_grad(*args)
            jax.block_until_ready(h_and_grad)
            value_grad_samples.append(_elapsed_ms(start_ns))

        mean_value = float(np.mean(value_samples))
        mean_value_grad = float(np.mean(value_grad_samples))

        print(
            "  first value call:      "
            f"{first_value_ms:.3f} ms (may include compilation)"
        )
        print(
            "  first value+grad call: "
            f"{first_value_grad_ms:.3f} ms (may include compilation)"
        )
        print(f"  value only:             {_stats_ms(value_samples)}")
        print(f"  value + gradient:       {_stats_ms(value_grad_samples)}")
        print(
            "  estimated AD overhead: "
            f"{mean_value_grad - mean_value:.3f} ms"
        )
        print(
            "  value+grad/value ratio: "
            f"{mean_value_grad / max(mean_value, 1e-12):.2f}x"
        )

    # ==============================================================
    # COMPUTE OPTIMAL COMMAND
    # ==============================================================

    def compute_command(self):
        profile_now = self._should_profile()
        total_start_ns = time.perf_counter_ns()

        start_ns = time.perf_counter_ns()
        acc_ref = self.compute_acceleration_reference()
        reference_ms = _elapsed_ms(start_ns)

        acceleration_lower_bound = -self.max_accel * np.ones(3, dtype=float)
        acceleration_upper_bound = self.max_accel * np.ones(3, dtype=float)

        constraint_rows = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        ]
        constraint_lower_bounds = list(acceleration_lower_bound)
        constraint_upper_bounds = list(acceleration_upper_bound)

        constraint_start_ns = time.perf_counter_ns()

        # Collect obstacles on the host before creating the fixed-size JAX batch.
        start_ns = time.perf_counter_ns()
        active_names: list[str] = []
        active_obstacles: list[dict[str, Any]] = []
        active_distances: list[float] = []

        for obstacle_name, obstacle in self.obstacles.items():
            distance = float(
                np.linalg.norm(
                    self.state[:3] - np.asarray(obstacle["p"], dtype=float)
                )
            )

            # The constraint is kept when overlapping: the CBF then only
            # forbids approaching the obstacle (see compute_candidate_h)
            if distance <= (
                self.collision_radius + float(obstacle["collision_radius"])
            ):
                print(
                    f"Overlapping {obstacle_name}: distance={distance:.3f}"
                )

            active_names.append(obstacle_name)
            active_obstacles.append(obstacle)
            active_distances.append(distance)

        filtering_ms = _elapsed_ms(start_ns)
        active_constraints = len(active_obstacles)

        state_h2d_ms = 0.0
        batch_h2d_ms = 0.0
        batch_h_grad_ms = 0.0
        batch_d2h_ms = 0.0
        cbf_algebra_ms = 0.0
        h_values = np.empty((0,), dtype=float)
        gradients = np.empty((0, 6), dtype=float)
        cbf_at_reference = np.empty((0,), dtype=float)

        if active_obstacles:
            start_ns = time.perf_counter_ns()
            state_device = jax.device_put(
                np.asarray(self.state, dtype=np.float64),
                self.jax_device,
            )
            state_device.block_until_ready()
            state_h2d_ms = _elapsed_ms(start_ns)

            obstacle_states_host = np.stack(
                [
                    np.concatenate((obstacle["p"], obstacle["v"]))
                    for obstacle in active_obstacles
                ],
                axis=0,
            ).astype(np.float64, copy=False)

            obstacle_radii_host = np.asarray(
                [
                    float(obstacle["collision_radius"])
                    for obstacle in active_obstacles
                ],
                dtype=np.float64,
            )

            start_ns = time.perf_counter_ns()
            obstacle_states_device = jax.device_put(
                obstacle_states_host,
                self.jax_device,
            )
            obstacle_radii_device = jax.device_put(
                obstacle_radii_host,
                self.jax_device,
            )
            jax.block_until_ready(
                (obstacle_states_device, obstacle_radii_device)
            )
            batch_h2d_ms = _elapsed_ms(start_ns)

            start_ns = time.perf_counter_ns()
            h_values_device, gradients_device = (
                _batched_candidate_h_value_and_grad(
                    state_device,
                    obstacle_states_device,
                    self.collision_radius,
                    obstacle_radii_device,
                    self.sh_n,
                    self.sh_tau,
                )
            )
            jax.block_until_ready((h_values_device, gradients_device))
            batch_h_grad_ms = _elapsed_ms(start_ns)

            start_ns = time.perf_counter_ns()
            h_values = np.asarray(h_values_device, dtype=float)
            gradients = np.asarray(gradients_device, dtype=float)
            batch_d2h_ms = _elapsed_ms(start_ns)

            start_ns = time.perf_counter_ns()
            class_k_values = 100.0 * h_values
            control_rows = gradients @ self.G

            # h depends on p_B - p_A, so L_f h = grad_h[:3] (v_A - v_B) for
            # obstacles moving at constant velocity
            relative_velocities = self.state[3:] - obstacle_states_host[:, 3:]

            drift_and_class_k = (
                np.einsum(
                    "ni,ni->n",
                    gradients[:, :3],
                    relative_velocities,
                )
                + class_k_values
            )

            cbf_lower_bounds = -drift_and_class_k
            cbf_at_reference = (
                drift_and_class_k + control_rows @ acc_ref
            )

            constraint_rows.extend(control_rows)
            constraint_lower_bounds.extend(cbf_lower_bounds.tolist())
            constraint_upper_bounds.extend(
                [np.inf] * active_constraints
            )
            cbf_algebra_ms = _elapsed_ms(start_ns)

        constraint_total_ms = _elapsed_ms(constraint_start_ns)

        start_ns = time.perf_counter_ns()
        lower = np.asarray(constraint_lower_bounds, dtype=float)
        upper = np.asarray(constraint_upper_bounds, dtype=float)
        A_qp = sparse.csc_matrix(np.vstack(constraint_rows), dtype=float)
        matrix_assembly_ms = _elapsed_ms(start_ns)

        start_ns = time.perf_counter_ns()
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.eye(3, format="csc", dtype=float),
            q=-acc_ref,
            A=A_qp,
            l=lower,
            u=upper,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=100,
        )
        osqp_setup_ms = _elapsed_ms(start_ns)

        start_ns = time.perf_counter_ns()
        results = solver.solve()
        osqp_solve_ms = _elapsed_ms(start_ns)

        start_ns = time.perf_counter_ns()
        status = results.info.status.lower()
        if results.x is not None and status.startswith("solved"):
            u_star = np.asarray(results.x, dtype=float)
            self.previous_solution = u_star.copy()
            self.cmd_accel = u_star.copy()
        else:
            self.cmd_accel = np.zeros(3, dtype=float)
        result_handling_ms = _elapsed_ms(start_ns)

        total_ms = _elapsed_ms(total_start_ns)

        if profile_now:
            print(
                f"\n[QP3D batched timing] step={self.step} "
                f"device={self.jax_device} "
                f"obstacles={len(self.obstacles)} "
                f"active={active_constraints}"
            )
            self._print_stage("acceleration reference", reference_ms)
            self._print_stage("filter/check obstacles", filtering_ms)
            self._print_stage("state host -> device", state_h2d_ms)
            self._print_stage("obstacle batch host -> device", batch_h2d_ms)
            self._print_stage("BATCH h+grad execute", batch_h_grad_ms)
            self._print_stage("batch result device -> host", batch_d2h_ms)
            self._print_stage("batched CBF host algebra", cbf_algebra_ms)

            for index, obstacle_name in enumerate(active_names):
                print(
                    f"  obstacle={obstacle_name} "
                    f"distance={active_distances[index]:.3f} "
                    f"h={h_values[index]:.6f} "
                    f"CBF(u_ref)={cbf_at_reference[index]:.6f}"
                )

            self._print_stage("all constraints", constraint_total_ms)
            self._print_stage("QP matrix assembly", matrix_assembly_ms)
            self._print_stage("OSQP setup", osqp_setup_ms)
            self._print_stage("OSQP solve", osqp_solve_ms)
            self._print_stage("result handling", result_handling_ms)
            self._print_stage("TOTAL compute_command", total_ms)
            print(
                f"  OSQP status={results.info.status}, "
                f"iterations={results.info.iter}, "
                f"run_time={results.info.run_time * 1e3:.3f} ms"
            )

        return self.cmd_accel
