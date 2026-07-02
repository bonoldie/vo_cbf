import time
import csv
import numpy as np
import mujoco
import mujoco.viewer


from controllers.mpc_sh import MPC_SH
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

target = np.array([3.0, 1.0])
target_side = 1
sh_degree = 6


def generate_new_target():
    global target, target_side
    target_side *= -1
    target = np.array([0.5 + target_side * np.random.uniform(1.0,1.5),  target_side * np.random.uniform(-0.5,-1.5)])
    controller.set_target(target)


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
            "pos": (-0.5, 2, 0.05)
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
    vl2, vr2 = vw_to_wheels(0.05, -0.5)

    d.ctrl[car2_left] = vl2
    d.ctrl[car2_right] = vr2

    # --------------------------------------------------
    # MPC
    # --------------------------------------------------

    x0, y0, yaw0 = get_2d_pose(
        d,
        bindings["car_1"]["bodies"]["car"]
    )

    controller = MPC_SH(
        target=target,
        initial_state=np.array([x0, y0, yaw0]),
        obstacles=get_collision_spheres(['car_1']),
        sh_degree=sh_degree
    )

    generate_new_target()

    while viewer.is_running():
        step_start = time.time()

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
        controller.update_obstacles(get_collision_spheres(['car_1']))

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

                for obstacle_name, obstacle in get_collision_spheres(['car_1']).items():

                    draw_sphere(
                        scene,
                        np.array(obstacle["p"]),
                        (0, 0, 1, 0.1),
                        obstacle["collision_radius"]
                    )

                    # obs_sh_params = controller.sh_params[obstacle_name] 

                    # obs_sh_params = MPC_SH.sh_params[obstacle_name]

                    # a_super = obs_sh_params["a"]
                    # b_super = obs_sh_params["b"]
                    # n = sh_degree

                    # center = np.asarray(obstacle["p"][:2])

                    # y_tan = np.linspace(-b_super, b_super, 20)

                    # for yt in y_tan:

                    #     xh = a_super * (1 + (yt / b_super) ** n) ** (1.0 / n)

                    #     # right branch
                    #     p = center + np.array([xh, yt])

                    #     draw_sphere(
                    #         scene,
                    #         np.array([p[0], p[1], 0.05]),
                    #         (0, 1, 0, 1),
                    #         0.01
                    #     )

                    #     # left branch (mirror)
                    #     p = center + np.array([-xh, yt])

                    #     draw_sphere(
                    #         scene,
                    #         np.array([p[0], p[1], 0.05]),
                    #         (0, 1, 0, 1),
                    #         0.01
                    #     )



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

        if step % 40 == 0:

            print(f"==============================")
            print(f"Sim time: {step * DT}s")
            print(
                f"[t={d.time:6.2f}] "
                f"x={x:6.2f} "
                f"y={y:6.2f} "
                f"dist={dist:6.3f} "
                f"v={v_cmd:6.3f}"
                f"w={w_cmd:6.3f}"
            )
            print(
                format_obstacles(get_collision_spheres())
            )

        if True:
            if abs(dist) <= 0.05:
                generate_new_target()
        else:
            target = np.array([pb.horizontal_offset * 0.01, pb.vertical_offset * 0.01])
            controller.set_target(target)

        step += 1

        time_until_next_step = DT - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

controller.stop()