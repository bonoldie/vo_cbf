import matplotlib.pyplot as plt
import numpy as np

class ControlWorldPlot:
    def __init__(
        self,
        R_control_sim: np.ndarray,
        axis_length: float = 0.5,
        plot_limit: float = 3.0,
    ):
        self.R_control_sim = np.asarray(R_control_sim, dtype=float)
        self.axis_length = axis_length

        if self.R_control_sim.shape != (3, 3):
            raise ValueError("R_control_sim must be a 3x3 matrix.")

        plt.ion()

        self.fig = plt.figure("Control-world frame")
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlabel("Control X")
        self.ax.set_ylabel("Control Y")
        self.ax.set_zlabel("Control Z")

        self.ax.set_xlim(-plot_limit, plot_limit)
        self.ax.set_ylim(-plot_limit, plot_limit)
        self.ax.set_zlim(-plot_limit, plot_limit)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.grid(True)

        # Fixed control-world frame at the origin.
        origin = np.zeros(3)

        self.ax.quiver(
            *origin,
            axis_length, 0.0, 0.0,
            color="r",
            arrow_length_ratio=0.15,
        )
        self.ax.quiver(
            *origin,
            0.0, axis_length, 0.0,
            color="g",
            arrow_length_ratio=0.15,
        )
        self.ax.quiver(
            *origin,
            0.0, 0.0, axis_length,
            color="b",
            arrow_length_ratio=0.15,
        )

        self.ax.text(axis_length, 0.0, 0.0, "Xc")
        self.ax.text(0.0, axis_length, 0.0, "Yc")
        self.ax.text(0.0, 0.0, axis_length, "Zc")

        # Robot position in control-world coordinates.
        self.robot_point, = self.ax.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="None",
            color="k",
            label="Drone",
        )

        # Drone body axes expressed in the control-world frame.
        self.robot_axes = [
            self.ax.plot([], [], [], color="r", linewidth=2)[0],
            self.ax.plot([], [], [], color="g", linewidth=2)[0],
            self.ax.plot([], [], [], color="b", linewidth=2)[0],
        ]

        # Robot trajectory in control-world coordinates.
        self.trajectory_line, = self.ax.plot(
            [],
            [],
            [],
            color="k",
            alpha=0.4,
        )

        self.trajectory = []

        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def _set_line_3d(line, start: np.ndarray, end: np.ndarray):
        line.set_data(
            [start[0], end[0]],
            [start[1], end[1]],
        )
        line.set_3d_properties(
            [start[2], end[2]]
        )

    def update(self, robot_state: np.ndarray):
        # Position expressed in simulation-world coordinates.
        position_sim = np.asarray(robot_state[:3], dtype=float)

        # Body orientation: body frame -> simulation-world frame.
        R_sim_body = np.asarray(
            robot_state[6:15],
            dtype=float,
        ).reshape(3, 3)

        # Transform position and orientation into control-world coordinates.
        position_control = self.R_control_sim @ position_sim
        R_control_body = self.R_control_sim @ R_sim_body

        self.robot_point.set_data(
            [position_control[0]],
            [position_control[1]],
        )
        self.robot_point.set_3d_properties(
            [position_control[2]]
        )

        # Columns of R_control_body are the body axes expressed in control world.
        for axis_index, line in enumerate(self.robot_axes):
            axis_end = (
                position_control
                + self.axis_length * R_control_body[:, axis_index]
            )

            self._set_line_3d(
                line,
                position_control,
                axis_end,
            )

        self.trajectory.append(position_control.copy())
        trajectory = np.asarray(self.trajectory)

        self.trajectory_line.set_data(
            trajectory[:, 0],
            trajectory[:, 1],
        )
        self.trajectory_line.set_3d_properties(
            trajectory[:, 2]
        )

        self.ax.set_title(
            "Drone in control-world frame\n"
            f"p = [{position_control[0]:.2f}, "
            f"{position_control[1]:.2f}, "
            f"{position_control[2]:.2f}]"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()