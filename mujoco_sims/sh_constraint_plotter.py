import math
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RobotPlotData:
    figure: Any
    axes: dict[str, Any]
    artists: dict[str, dict[str, Any]]
    obstacle_names: tuple[str, ...]


class SHConstraintPlotter:
    """
    Plot the 3D SH-CBF as a 2D section in relative-velocity space.

    Horizontal axis:
        tangential relative speed ||v_perpendicular||

    Vertical axis:
        radial relative speed v_parallel

    The SH-CBF is:

        h = a * (1 + (v_tangential / b)^n)^(1/n) - v_parallel

    Therefore the safe region h >= 0 is below the SH boundary.
    """

    def __init__(
        self,
        n: int,
        gamma: float = 100.0,
        beta: float = 0.0,
        plot_every: int = 5,
        minimum_axis_range: float = 0.25,
    ):
        self.n = int(n)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.plot_every = max(1, int(plot_every))
        self.minimum_axis_range = float(minimum_axis_range)

        self._robots: dict[str, RobotPlotData] = {}
        self._update_counter: dict[str, int] = {}

        plt.ion()

    @staticmethod
    def _velocity_coordinates(
        robot_state: np.ndarray,
        obstacle_state: dict,
    ) -> tuple[float, float, np.ndarray]:
        """
        Return:
            v_tangential
            v_parallel
            line-of-sight unit vector
        """

        robot_state = np.asarray(robot_state, dtype=np.float64)

        p_robot = robot_state[:3]
        v_robot = robot_state[3:]

        p_obstacle = np.asarray(
            obstacle_state["p"],
            dtype=np.float64,
        )

        v_obstacle = np.asarray(
            obstacle_state["v"],
            dtype=np.float64,
        )

        delta_p = p_obstacle - p_robot
        distance = np.linalg.norm(delta_p)

        if distance < 1e-12:
            e_los = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            )
        else:
            e_los = delta_p / distance

        v_relative = v_robot - v_obstacle

        v_parallel = float(
            np.dot(v_relative, e_los)
        )

        v_perpendicular = (
            v_relative
            - v_parallel * e_los
        )

        v_tangential = float(
            np.linalg.norm(v_perpendicular)
        )

        return v_tangential, v_parallel, e_los

    @staticmethod
    def _commanded_velocity_coordinates(
        robot_state: np.ndarray,
        obstacle_state: dict,
        acceleration_command: np.ndarray,
        obstacle_acceleration: np.ndarray,
        dt: float,
        e_los: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute the one-step relative velocity produced by the selected
        robot acceleration.

        The SH boundary is kept at the current position, so this shows
        only the immediate velocity-space effect of the command.
        """

        robot_state = np.asarray(
            robot_state,
            dtype=np.float64,
        )

        acceleration_command = np.asarray(
            acceleration_command,
            dtype=np.float64,
        )

        obstacle_acceleration = np.asarray(
            obstacle_acceleration,
            dtype=np.float64,
        )

        v_robot = robot_state[3:]

        v_obstacle = np.asarray(
            obstacle_state["v"],
            dtype=np.float64,
        )

        next_relative_velocity = (
            v_robot
            - v_obstacle
            + dt
            * (
                acceleration_command
                - obstacle_acceleration
            )
        )

        next_v_parallel = float(
            np.dot(next_relative_velocity, e_los)
        )

        next_v_perpendicular = (
            next_relative_velocity
            - next_v_parallel * e_los
        )

        next_v_tangential = float(
            np.linalg.norm(next_v_perpendicular)
        )

        return next_v_tangential, next_v_parallel

    def _ensure_figure(
        self,
        robot_name: str,
        obstacle_names: tuple[str, ...],
    ) -> RobotPlotData:
        existing = self._robots.get(robot_name)

        if (
            existing is not None
            and existing.obstacle_names == obstacle_names
        ):
            return existing

        if existing is not None:
            plt.close(existing.figure)

        number_of_obstacles = len(obstacle_names)

        number_of_columns = max(
            1,
            math.ceil(math.sqrt(number_of_obstacles)),
        )

        number_of_rows = max(
            1,
            math.ceil(
                number_of_obstacles
                / number_of_columns
            ),
        )

        figure, axes_array = plt.subplots(
            number_of_rows,
            number_of_columns,
            figsize=(
                5.0 * number_of_columns,
                4.2 * number_of_rows,
            ),
            squeeze=False,
        )

        flat_axes = axes_array.reshape(-1)

        axes: dict[str, Any] = {}
        artists: dict[str, dict[str, Any]] = {}

        for index, obstacle_name in enumerate(obstacle_names):
            axis = flat_axes[index]

            boundary_line, = axis.plot(
                [],
                [],
                linewidth=2.0,
                label="SH boundary",
            )

            current_point, = axis.plot(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                label="current velocity",
            )

            commanded_point, = axis.plot(
                [],
                [],
                marker="x",
                linestyle="none",
                markersize=9,
                markeredgewidth=2,
                label="after command",
            )

            command_segment, = axis.plot(
                [],
                [],
                linestyle="--",
                linewidth=1.5,
                label="command effect",
            )

            axis.axvline(
                0.0,
                linewidth=0.8,
                linestyle=":",
            )

            axis.axhline(
                0.0,
                linewidth=0.8,
                linestyle=":",
            )

            information_text = axis.text(
                0.02,
                0.98,
                "",
                transform=axis.transAxes,
                verticalalignment="top",
                family="monospace",
            )

            axis.set_xlabel(
                r"$v_{\mathrm{tangential}}$ [m/s]"
            )

            axis.set_ylabel(
                r"$v_{\parallel}$ [m/s]"
            )

            axis.grid(True)
            axis.legend(loc="lower right")

            axes[obstacle_name] = axis

            artists[obstacle_name] = {
                "boundary": boundary_line,
                "current": current_point,
                "commanded": commanded_point,
                "command_segment": command_segment,
                "text": information_text,
                "safe_fill": None,
            }

        for index in range(
            number_of_obstacles,
            len(flat_axes),
        ):
            flat_axes[index].set_visible(False)

        figure.suptitle(
            f"SH-CBF relative-velocity view: {robot_name}"
        )

        figure.tight_layout()

        result = RobotPlotData(
            figure=figure,
            axes=axes,
            artists=artists,
            obstacle_names=obstacle_names,
        )

        self._robots[robot_name] = result

        return result

    def update(
        self,
        robot_name: str,
        robot_state: np.ndarray,
        obstacle_states: dict,
        boundaries: dict,
        acceleration_command: np.ndarray,
        dt: float,
        obstacle_accelerations: dict | None = None,
    ) -> None:
        """
        Update the figure associated with one robot.

        obstacle_states must use the same structure used by the controller:

            {
                obstacle_name: {
                    "p": np.array([x, y, z]),
                    "v": np.array([vx, vy, vz]),
                    ...
                }
            }

        boundaries must contain:

            {
                obstacle_name: {
                    "a": ...,
                    "b": ...,
                    "h_value": ...,
                    "grad_h_value": np.ndarray(6),
                }
            }
        """

        counter = self._update_counter.get(robot_name, 0) + 1
        self._update_counter[robot_name] = counter

        if counter % self.plot_every != 0:
            return

        obstacle_names = tuple(
            sorted(
                set(boundaries.keys())
                & set(obstacle_states.keys())
            )
        )

        if not obstacle_names:
            return

        plot_data = self._ensure_figure(
            robot_name,
            obstacle_names,
        )

        robot_state = np.asarray(
            robot_state,
            dtype=np.float64,
        )

        acceleration_command = np.asarray(
            acceleration_command,
            dtype=np.float64,
        )

        for obstacle_name in obstacle_names:
            boundary = boundaries[obstacle_name]
            obstacle = obstacle_states[obstacle_name]

            axis = plot_data.axes[obstacle_name]
            artist = plot_data.artists[obstacle_name]

            a = float(
                np.asarray(boundary["a"]).item()
            )

            b = float(
                np.asarray(boundary["b"]).item()
            )

            h_value = float(
                np.asarray(
                    boundary["h_value"]
                ).item()
            )

            grad_h = np.asarray(
                boundary["grad_h_value"],
                dtype=np.float64,
            ).reshape(6)

            b_safe = max(abs(b), 1e-12)

            (
                current_v_tangential,
                current_v_parallel,
                e_los,
            ) = self._velocity_coordinates(
                robot_state,
                obstacle,
            )

            if obstacle_accelerations is None:
                obstacle_acceleration = np.zeros(
                    3,
                    dtype=np.float64,
                )
            else:
                obstacle_acceleration = np.asarray(
                    obstacle_accelerations.get(
                        obstacle_name,
                        np.zeros(3),
                    ),
                    dtype=np.float64,
                )

            (
                commanded_v_tangential,
                commanded_v_parallel,
            ) = self._commanded_velocity_coordinates(
                robot_state=robot_state,
                obstacle_state=obstacle,
                acceleration_command=acceleration_command,
                obstacle_acceleration=obstacle_acceleration,
                dt=dt,
                e_los=e_los,
            )

            x_limit = max(
                self.minimum_axis_range,
                3.0 * b_safe,
                1.4 * current_v_tangential,
                1.4 * commanded_v_tangential,
            )

            x_values = np.linspace(
                -x_limit,
                x_limit,
                500,
            )

            boundary_values = (
                a
                * (
                    1.0
                    + (
                        np.abs(x_values)
                        / b_safe
                    ) ** self.n
                ) ** (1.0 / self.n)
            )

            current_boundary_value = (
                a
                * (
                    1.0
                    + (
                        current_v_tangential
                        / b_safe
                    ) ** self.n
                ) ** (1.0 / self.n)
            )

            commanded_boundary_value = (
                a
                * (
                    1.0
                    + (
                        commanded_v_tangential
                        / b_safe
                    ) ** self.n
                ) ** (1.0 / self.n)
            )

            commanded_h_value = (
                commanded_boundary_value
                - commanded_v_parallel
            )

            # ------------------------------------------------------
            # Correct continuous-time CBF residual
            # for a moving obstacle.
            #
            # hdot =
            #   grad_position @ (v_robot - v_obstacle)
            #   + grad_velocity @ (u_robot - u_obstacle)
            # ------------------------------------------------------

            robot_velocity = robot_state[3:]

            obstacle_velocity = np.asarray(
                obstacle["v"],
                dtype=np.float64,
            )

            relative_velocity = (
                robot_velocity - obstacle_velocity
            )

            relative_acceleration = (
                acceleration_command
                - obstacle_acceleration
            )

            alpha_h = (
                self.gamma * h_value
                + self.beta * h_value**3
            )

            cbf_residual = float(
                grad_h[:3] @ relative_velocity
                + grad_h[3:] @ relative_acceleration
                + alpha_h
            )

            y_min = min(
                -0.15,
                current_v_parallel,
                commanded_v_parallel,
                float(np.min(boundary_values)),
            )

            y_max = max(
                0.15,
                current_v_parallel,
                commanded_v_parallel,
                float(np.max(boundary_values)),
            )

            y_margin = max(
                0.05,
                0.12 * (y_max - y_min),
            )

            y_min -= y_margin
            y_max += y_margin

            artist["boundary"].set_data(
                x_values,
                boundary_values,
            )

            artist["current"].set_data(
                [current_v_tangential],
                [current_v_parallel],
            )

            artist["commanded"].set_data(
                [commanded_v_tangential],
                [commanded_v_parallel],
            )

            artist["command_segment"].set_data(
                [
                    current_v_tangential,
                    commanded_v_tangential,
                ],
                [
                    current_v_parallel,
                    commanded_v_parallel,
                ],
            )

            old_fill = artist["safe_fill"]

            if old_fill is not None:
                old_fill.remove()

            artist["safe_fill"] = axis.fill_between(
                x_values,
                y_min,
                boundary_values,
                alpha=0.12,
                label=None,
            )

            artist["text"].set_text(
                f"a       = {a:+.5f}\n"
                f"b       = {b:+.5f}\n"
                f"h       = {h_value:+.5e}\n"
                f"h next  = {commanded_h_value:+.5e}\n"
                f"CBF     = {cbf_residual:+.5e}\n"
                f"vt      = {current_v_tangential:+.5f}\n"
                f"vpar    = {current_v_parallel:+.5f}"
            )

            axis.set_xlim(
                -x_limit,
                x_limit,
            )

            axis.set_ylim(
                y_min,
                y_max,
            )

            # Red title means either already unsafe or the selected
            # command violates the continuous-time CBF condition.
            unsafe = (
                h_value < 0.0
                or cbf_residual < 0.0
            )

            axis.set_title(
                obstacle_name,
                color="red" if unsafe else "black",
            )

        plot_data.figure.canvas.draw_idle()
        plot_data.figure.canvas.flush_events()