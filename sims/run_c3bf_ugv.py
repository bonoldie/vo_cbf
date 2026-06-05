import time
import csv
import numpy as np
import mujoco
import mujoco.viewer

from controllers.simple_mpc import SimpleMPC
from generate_scenarios import buildModel, format_obstacles
from utils.playback import Playback
from utils.scenebuilder import ObstacleType
from utils.utils import (
    draw_sphere,
    draw_vector,
    get_2d_pose,
    wrap
)

# --------------------------------------------------
# Scenario
# --------------------------------------------------

target = np.array([1.0, 1.0])

obstacles = [
    {
        "type": ObstacleType.BOX,
        "pos": (1, 0, 0.1),
        "size": (0.1, 0.1, 0.1)
    },
    {
        "type": ObstacleType.CYLINDER,
        "pos": (0, 1, 0.05),
        "radius": 0.25
    }
]

m, d, bindings, get_collision_spheres = buildModel(
    [
        {
            "name": "car_1",
            "collision_radius": 0.14,
            "pos": (-1, -2, 0.05)
        },
        {
            "name": "car_2",
            "collision_radius": 0.14,
            "pos": (0, -0.35, 0.05)
        }
    ],
    obstacles
)

# --------------------------------------------------
# Robot
# --------------------------------------------------

wheel_radius = 0.015
wheel_base = 0.02

left_actuator_id = bindings["car_1"]["actuators"]["left_wheel"]
right_actuator_id = bindings["car_1"]["actuators"]["right_wheel"]

car2_left = bindings["car_2"]["actuators"]["left_wheel"]
car2_right = bindings["car_2"]["actuators"]["right_wheel"]

DT = m.opt.timestep

# --------------------------------------------------
# MPC
# --------------------------------------------------

x0, y0, yaw0 = get_2d_pose(
    d,
    bindings["car_1"]["bodies"]["car"]
)

controller = SimpleMPC(
    target=target,
    initial_state=np.array([x0, y0, yaw0])
)

# --------------------------------------------------
# Helper
# --------------------------------------------------

def vw_to_wheels(v, w):
    vl = (v - 0.5 * wheel_base * w) / wheel_radius
    vr = (v + 0.5 * wheel_base * w) / wheel_radius

    return (
        np.clip(vl, -15, 15),
        np.clip(vr, -15, 15)
    )

# --------------------------------------------------
# Logging
# --------------------------------------------------

log = []

# --------------------------------------------------
# Viewer loop
# --------------------------------------------------

with mujoco.viewer.launch_passive(
    m,
    d,
    show_left_ui=False,
    show_right_ui=False
) as viewer:

    pb = Playback()
    step = 0

    # moving obstacle
    vl2, vr2 = vw_to_wheels(0.15, 3.0)

    d.ctrl[car2_left] = vl2
    d.ctrl[car2_right] = vr2

    while viewer.is_running():

        if pb.paused:
            time.sleep(0.1)
            continue

        # ------------------------------------------
        # State
        # ------------------------------------------

        x, y, yaw = get_2d_pose(
            d,
            bindings["car_1"]["bodies"]["car"]
        )

        controller.update_state(x, y, yaw)

        # ------------------------------------------
        # Control
        # ------------------------------------------

        v_cmd, w_cmd = controller.get_command()

        vl, vr = vw_to_wheels(v_cmd, w_cmd)

        d.ctrl[left_actuator_id] = vl
        d.ctrl[right_actuator_id] = vr

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        pos = np.array([x, y])

        dist = np.linalg.norm(target - pos)

        desired_yaw = np.arctan2(
            target[1] - y,
            target[0] - x
        )

        yaw_err = wrap(desired_yaw - yaw)

        # ------------------------------------------
        # Visualization
        # ------------------------------------------

        with viewer.lock():

            scene = viewer.user_scn
            scene.ngeom = 0

            pos3 = np.array([x, y, 0.05])

            forward = np.array([
                np.cos(yaw),
                np.sin(yaw),
                0.0
            ])

            draw_vector(
                scene,
                pos3,
                forward * v_cmd * 10,
                [1, 1, 0, 0.8]
            )

            draw_vector(
                scene,
                pos3,
                np.array([0, 0, w_cmd]),
                [1, 0, 0, 0.8]
            )

            draw_sphere(
                scene,
                np.append(target, 0.05),
                (1, 0, 0, 1),
                0.03
            )

            if pb.show_obstacles_collision_boxes:

                for obstacle in get_collision_spheres().values():

                    draw_sphere(
                        scene,
                        np.array(obstacle["p"]),
                        (0, 0, 1, 0.1),
                        obstacle["collision_radius"]
                    )

        mujoco.mj_step(m, d)
        viewer.sync()

        # ------------------------------------------
        # Logging
        # ------------------------------------------

        log.append([
            d.time,
            x,
            y,
            yaw,
            v_cmd,
            w_cmd,
            dist,
            yaw_err
        ])

        if step % 20 == 0:

            print(
                f"[t={d.time:6.2f}] "
                f"x={x:6.2f} "
                f"y={y:6.2f} "
                f"dist={dist:6.3f} "
                f"v={v_cmd:6.3f} "
                f"w={w_cmd:6.3f}"
            )

        step += 1

controller.stop()