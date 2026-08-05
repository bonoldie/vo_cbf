import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class RelativeDistancePlotter:
    """
    Real-time plot of the Euclidean distance between one reference robot
    and every other robot.

    It also periodically saves:
        - relative_distances.csv
        - relative_distances.png
    """

    def __init__(
        self,
        reference_robot: str,
        output_directory: str = "relative_distance_results",
        plot_every: int = 2,
        save_every: int = 100,
    ):
        self.reference_robot = reference_robot
        self.plot_every = max(1, int(plot_every))
        self.save_every = max(1, int(save_every))

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = (
            self.output_directory
            / "relative_distances.csv"
        )

        self.plot_path = (
            self.output_directory
            / "relative_distances.png"
        )

        self.times: list[float] = []
        self.distances: dict[str, list[float]] = {}

        self.lines = {}
        self.threshold_lines = {}

        self.update_counter = 0

        plt.ion()

        self.figure, self.axis = plt.subplots(
            figsize=(10, 5),
        )

        self.axis.set_title(
            f"Relative distances from {reference_robot}"
        )

        self.axis.set_xlabel("Simulation time [s]")
        self.axis.set_ylabel("Relative distance [m]")
        self.axis.grid(True)

        self.figure.tight_layout()

    def update(
        self,
        simulation_time: float,
        state: dict,
        collision_radii: dict[str, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        simulation_time:
            Current MuJoCo simulation time.

        state:
            Dictionary containing:

                state[robot_name]["position"]

        collision_radii:
            Optional dictionary:

                {
                    "robot0": 0.15,
                    "robot1": 0.15,
                    ...
                }

            A horizontal line at the combined collision radius will be
            drawn for every robot pair.
        """

        if self.reference_robot not in state:
            return

        reference_position = np.asarray(
            state[self.reference_robot]["position"],
            dtype=np.float64,
        )

        other_robots = sorted(
            name
            for name in state
            if name != self.reference_robot
        )

        if not other_robots:
            return

        self.times.append(float(simulation_time))

        # Add one empty sample to all existing histories.
        for history in self.distances.values():
            history.append(np.nan)

        for other_robot in other_robots:
            other_position = np.asarray(
                state[other_robot]["position"],
                dtype=np.float64,
            )

            relative_distance = float(
                np.linalg.norm(
                    other_position - reference_position
                )
            )

            # Handle robots that appear after plotting has started.
            if other_robot not in self.distances:
                self.distances[other_robot] = (
                    [np.nan] * len(self.times)
                )

                line, = self.axis.plot(
                    [],
                    [],
                    linewidth=2,
                    label=(
                        f"{self.reference_robot}"
                        f" → {other_robot}"
                    ),
                )

                self.lines[other_robot] = line

            self.distances[other_robot][-1] = (
                relative_distance
            )

            # Optional collision-distance threshold.
            if (
                collision_radii is not None
                and other_robot not in self.threshold_lines
                and self.reference_robot in collision_radii
                and other_robot in collision_radii
            ):
                minimum_distance = (
                    collision_radii[self.reference_robot]
                    + collision_radii[other_robot]
                )

                threshold_line = self.axis.axhline(
                    minimum_distance,
                    linestyle="--",
                    linewidth=1,
                    label=(
                        f"collision threshold "
                        f"{self.reference_robot}"
                        f"–{other_robot}: "
                        f"{minimum_distance:.3f} m"
                    ),
                )

                self.threshold_lines[other_robot] = (
                    threshold_line
                )

        self.update_counter += 1

        if self.update_counter % self.plot_every == 0:
            self._update_plot()

        if self.update_counter % self.save_every == 0:
            self.save()

    def _update_plot(self) -> None:
        time_array = np.asarray(
            self.times,
            dtype=np.float64,
        )

        for robot_name, line in self.lines.items():
            distance_array = np.asarray(
                self.distances[robot_name],
                dtype=np.float64,
            )

            line.set_data(
                time_array,
                distance_array,
            )

        self.axis.relim()
        self.axis.autoscale_view()

        # Distance cannot be negative.
        current_bottom, current_top = (
            self.axis.get_ylim()
        )

        self.axis.set_ylim(
            bottom=0.0,
            top=max(current_top, 0.1),
        )

        # Avoid recreating the legend every simulation step.
        if not hasattr(self, "_legend_created"):
            self.axis.legend(loc="best")
            self._legend_created = True

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def save(self) -> None:
        """Save the current data and figure."""

        robot_names = sorted(self.distances.keys())

        with self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(
                ["time"]
                + [
                    (
                        f"distance_"
                        f"{self.reference_robot}_"
                        f"{robot_name}"
                    )
                    for robot_name in robot_names
                ]
            )

            for index, simulation_time in enumerate(
                self.times
            ):
                writer.writerow(
                    [simulation_time]
                    + [
                        self.distances[robot_name][index]
                        for robot_name in robot_names
                    ]
                )

        self.figure.savefig(
            self.plot_path,
            dpi=200,
            bbox_inches="tight",
        )

    def close(self) -> None:
        """Save final results and close the figure."""

        self._update_plot()
        self.save()
        plt.close(self.figure)