import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d

class ControlWorldPlot:
    def __init__(
        self,
        R_control_sim: np.ndarray,
        axis_length: float = 0.5,
        plot_limit: float = 3.0,
        max_trajectory_points: int = 5000,
        rotors = []
    ):
        """
        Parameters
        ----------
        R_control_sim:
            Rotation matrix such that

                v_control = R_control_sim @ v_sim

        axis_length:
            Length used to draw body-frame axes.

        plot_limit:
            Limits applied to all plot axes.

        max_trajectory_points:
            Maximum number of stored trajectory samples.
        """
        self.R_control_sim = np.asarray(R_control_sim, dtype=float)
        self.axis_length = float(axis_length)
        self.max_trajectory_points = int(max_trajectory_points)
        self.rotors = rotors

        if self.R_control_sim.shape != (3, 3):
            raise ValueError("R_control_sim must be a 3x3 matrix.")

        if not np.allclose(
            self.R_control_sim.T @ self.R_control_sim,
            np.eye(3),
            atol=1e-8,
        ):
            raise ValueError("R_control_sim must be orthonormal.")

        if not np.isclose(
            np.linalg.det(self.R_control_sim),
            1.0,
            atol=1e-8,
        ):
            raise ValueError("R_control_sim must have determinant +1.")

        plt.ion()

        self.fig = plt.figure(
            "Simulation and control-world comparison",
            figsize=(14, 7),
        )

        self.ax_sim = self.fig.add_subplot(
            121,
            projection="3d",
        )

        self.ax_control = self.fig.add_subplot(
            122,
            projection="3d",
        )

        self._configure_axis(
            self.ax_sim,
            title="Simulation world",
            labels=("Simulation X", "Simulation Y", "Simulation Z"),
            frame_labels=("Xs", "Ys", "Zs"),
            plot_limit=plot_limit,
        )

        self._configure_axis(
            self.ax_control,
            title="Aligned control world",
            labels=("Control X", "Control Y", "Control Z"),
            frame_labels=("Xc", "Yc", "Zc"),
            plot_limit=plot_limit,
        )

        # ----------------------------------------------------------
        # Robot markers
        # ----------------------------------------------------------

        self.sim_robot_point, = self.ax_sim.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="None",
            color="k",
            label="Drone centre",
        )

        self.control_robot_point, = self.ax_control.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="None",
            color="k",
            label="Drone centre",
        )
        
        
        self.sim_rotors_point = []
        self.control_rotors_point = []
        
        self.sim_rotor_texts = []
        self.control_rotor_texts = []
        
        for i in range(len(self.rotors)):
            sim_rotor_point, = self.ax_sim.plot([], [], [], marker="o", linestyle="None", color="k", label=f"sim rotor {i+1}")
            control_rotor_point, = self.ax_control.plot([], [], [], marker="o", linestyle="None", color="k", label=f"control rotor {i+1}")
            self.sim_rotors_point.append(sim_rotor_point)
            self.control_rotors_point.append(control_rotor_point)
            
            self.sim_rotor_texts.append(self.ax_sim.text2D(0.0,0.0, f'rotor {i+1}'))
            self.control_rotor_texts.append(self.ax_control.text2D(0.0,0.0, f'rotor {i+1}'))
            

        # ----------------------------------------------------------
        # Body axes
        #
        # Red   = body x
        # Green = body y
        # Blue  = body z
        # ----------------------------------------------------------

        self.sim_body_axes = self._create_body_axes(self.ax_sim)
        self.control_body_axes = self._create_body_axes(self.ax_control)

        # ----------------------------------------------------------
        # Trajectories
        # ----------------------------------------------------------

        self.sim_trajectory_line, = self.ax_sim.plot(
            [],
            [],
            [],
            color="k",
            alpha=0.35,
            label="Trajectory",
        )

        self.control_trajectory_line, = self.ax_control.plot(
            [],
            [],
            [],
            color="k",
            alpha=0.35,
            label="Trajectory",
        )

        self.sim_trajectory = []
        self.control_trajectory = []

        # ----------------------------------------------------------
        # Optional rotor-1 visualization
        # ----------------------------------------------------------

        self.sim_rotor1_point, = self.ax_sim.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="None",
            color="m",
            label="Rotor 1",
        )

        self.control_rotor1_point, = self.ax_control.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="None",
            color="m",
            label="Rotor 1",
        )

        self.sim_rotor1_arm, = self.ax_sim.plot(
            [],
            [],
            [],
            color="m",
            linewidth=2,
        )

        self.control_rotor1_arm, = self.ax_control.plot(
            [],
            [],
            [],
            color="m",
            linewidth=2,
        )

        self.ax_sim.legend()
        self.ax_control.legend()

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _configure_axis(
        self,
        ax,
        title: str,
        labels: tuple[str, str, str],
        frame_labels: tuple[str, str, str],
        plot_limit: float,
    ):
        ax.set_title(title)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.set_zlim(-plot_limit, plot_limit)

        ax.set_box_aspect((1, 1, 1))
        ax.grid(True)

        origin = np.zeros(3)

        ax.quiver(
            *origin,
            self.axis_length,
            0.0,
            0.0,
            color="r",
            arrow_length_ratio=0.15,
        )

        ax.quiver(
            *origin,
            0.0,
            self.axis_length,
            0.0,
            color="g",
            arrow_length_ratio=0.15,
        )

        ax.quiver(
            *origin,
            0.0,
            0.0,
            self.axis_length,
            color="b",
            arrow_length_ratio=0.15,
        )

        ax.text(
            self.axis_length,
            0.0,
            0.0,
            frame_labels[0],
        )

        ax.text(
            0.0,
            self.axis_length,
            0.0,
            frame_labels[1],
        )

        ax.text(
            0.0,
            0.0,
            self.axis_length,
            frame_labels[2],
        )

    @staticmethod
    def _create_body_axes(ax):
        return [
            ax.plot(
                [],
                [],
                [],
                color="r",
                linewidth=2,
                label="Body X",
            )[0],
            ax.plot(
                [],
                [],
                [],
                color="g",
                linewidth=2,
                label="Body Y",
            )[0],
            ax.plot(
                [],
                [],
                [],
                color="b",
                linewidth=2,
                label="Body Z",
            )[0],
        ]

    @staticmethod
    def _set_point_3d(point, position: np.ndarray):
        point.set_data(
            [position[0]],
            [position[1]],
        )
        point.set_3d_properties(
            [position[2]]
        )

    @staticmethod
    def _set_line_3d(
        line,
        start: np.ndarray,
        end: np.ndarray,
    ):
        line.set_data(
            [start[0], end[0]],
            [start[1], end[1]],
        )

        line.set_3d_properties(
            [start[2], end[2]]
        )

    @staticmethod
    def _set_trajectory_3d(
        line,
        trajectory: list[np.ndarray],
    ):
        if not trajectory:
            return

        trajectory_array = np.asarray(trajectory)

        line.set_data(
            trajectory_array[:, 0],
            trajectory_array[:, 1],
        )

        line.set_3d_properties(
            trajectory_array[:, 2]
        )

    def _update_body_axes(
        self,
        lines,
        position: np.ndarray,
        orientation: np.ndarray,
    ):
        # Each column is a body axis expressed in the selected world frame.
        for axis_index, line in enumerate(lines):
            axis_direction = orientation[:, axis_index]

            axis_end = (
                position
                + self.axis_length * axis_direction
            )

            self._set_line_3d(
                line,
                position,
                axis_end,
            )

    def update(
        self,
        robot_state: np.ndarray,
        rotor1_position_sim: np.ndarray | None = None,
    ):
        robot_state = np.asarray(robot_state, dtype=float)

        if robot_state.size < 15:
            raise ValueError(
                "robot_state must contain at least position, velocity "
                "and a flattened 3x3 orientation matrix."
            )

        # ----------------------------------------------------------
        # Simulation-world pose
        # ----------------------------------------------------------

        position_sim = robot_state[:3]

        # Expected convention:
        #
        #     v_sim = R_sim_body @ v_body
        #
        R_sim_body = robot_state[6:15].reshape(3, 3)

        # ----------------------------------------------------------
        # Control-world pose
        # ----------------------------------------------------------

        position_control = (
            self.R_control_sim
            @ position_sim
        )
        
        # position_control[2] = -position_control[2]

        
          

        # ----------------------------------------------------------
        # Robot markers
        # ----------------------------------------------------------

        self._set_point_3d(
            self.sim_robot_point,
            position_sim,
        )

        self._set_point_3d(
            self.control_robot_point,
            position_control,
        )

        # ----------------------------------------------------------
        # Body axes
        # ----------------------------------------------------------

        self._update_body_axes(
            self.sim_body_axes,
            position_sim,
            R_sim_body,
        )

        self._update_body_axes(
            self.control_body_axes,
            position_control,
            R_sim_body,
        )
        
        # ----------------------------------------------------------
        # Rotors
        # ----------------------------------------------------------
        
        R_control_body = (
            self.R_control_sim @
            R_sim_body
        )
        
        for i, rotor_position in enumerate(self.rotors):    
            sim_rotor_position = position_sim + R_sim_body @ rotor_position
            control_rotor_position = position_control + R_control_body @ rotor_position
            
            self._set_point_3d(
                self.sim_rotors_point[i],
                sim_rotor_position,
            )
            
            self._set_point_3d(
                self.control_rotors_point[i],
                control_rotor_position,
            )
            
            sim_x, sim_y, _ = proj3d.proj_transform(sim_rotor_position[0], sim_rotor_position[1], sim_rotor_position[2], self.ax_sim.get_proj())
            control_x, control_y, _ = proj3d.proj_transform(control_rotor_position[0], control_rotor_position[1], control_rotor_position[2], self.ax_control.get_proj())

            self.sim_rotor_texts[i].set_position((sim_x, sim_y))
            self.control_rotor_texts[i].set_position((control_x, control_y))
        
        # ----------------------------------------------------------
        # Trajectories
        # ----------------------------------------------------------

        self.sim_trajectory.append(position_sim.copy())
        self.control_trajectory.append(position_control.copy())

        if len(self.sim_trajectory) > self.max_trajectory_points:
            self.sim_trajectory.pop(0)
            self.control_trajectory.pop(0)

        self._set_trajectory_3d(
            self.sim_trajectory_line,
            self.sim_trajectory,
        )

        self._set_trajectory_3d(
            self.control_trajectory_line,
            self.control_trajectory,
        )

        # ----------------------------------------------------------
        # Rotor 1
        # ----------------------------------------------------------

        rotor_text = ""

        if rotor1_position_sim is not None:
            rotor1_position_sim = np.asarray(
                rotor1_position_sim,
                dtype=float,
            )

            rotor1_position_control = (
                self.R_control_sim
                @ rotor1_position_sim
            )

            self._set_point_3d(
                self.sim_rotor1_point,
                rotor1_position_sim,
            )

            self._set_point_3d(
                self.control_rotor1_point,
                rotor1_position_control,
            )

            self._set_line_3d(
                self.sim_rotor1_arm,
                position_sim,
                rotor1_position_sim,
            )

            self._set_line_3d(
                self.control_rotor1_arm,
                position_control,
                rotor1_position_control,
            )

            rotor1_relative_control = (
                rotor1_position_control
                - position_control
            )

            rotor_angle_control = np.rad2deg(
                np.arctan2(
                    rotor1_relative_control[1],
                    rotor1_relative_control[0],
                )
            )

            rotor_text = (
                "\n"
                f"rotor 1 relative = "
                f"[{rotor1_relative_control[0]:.3f}, "
                f"{rotor1_relative_control[1]:.3f}, "
                f"{rotor1_relative_control[2]:.3f}]"
                "\n"
                f"rotor 1 XY angle = {rotor_angle_control:.2f}°"
            )

        # ----------------------------------------------------------
        # Titles
        # ----------------------------------------------------------

        self.ax_sim.set_title(
            "Simulation world\n"
            f"p_sim = [{position_sim[0]:.2f}, "
            f"{position_sim[1]:.2f}, "
            f"{position_sim[2]:.2f}]"
        )

        self.ax_control.set_title(
            "Aligned control world\n"
            f"p_control = [{position_control[0]:.2f}, "
            f"{position_control[1]:.2f}, "
            f"{position_control[2]:.2f}]"
            f"{rotor_text}"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()