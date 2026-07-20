import time
import csv
import numpy as np
import mujoco
import mujoco.viewer


from generate_scenarios import buildModel, format_obstacles
from utils.playback import Playback
from utils.scenebuilder import ObstacleType
from utils.utils import (
    draw_sphere,
    draw_vector,
    get_3d_position,
    get_3d_velocity
)

from controllers.qp_3d import QP3D

# --------------------------------------------------
# Scenario
# --------------------------------------------------
target = np.array([1.0, 0.0, 0.0])

# SH params
sh_n = 6
sh_tau = 1.2

# Control params
ref_speed = 0.2
max_accel = 2.0

controller = None

def generate_new_target():
    global target, controller
    target = np.array([np.random.uniform(-2.0,2.0), np.random.uniform(-2.0,2.0),np.random.uniform(0.2,4.0)])
    controller.set_target(target)


obstacles = [
    {
        "type": ObstacleType.SPHERE,
        "pos": (1, 1, 1),
        "radius": 0.25,
        
    },
    {
        "type": ObstacleType.SPHERE,
        "pos": (2, 2, 2),
        "radius": 0.25
    }
]

m, d, bindings, get_collision_spheres = buildModel(
    [
        {
            "name": "robot1",
            "collision_radius": 0.15,
            "pos": (0, 0, 0),
            "robot_path": "scenarios/double_integrator/bot.xml"
        }
    ],
    obstacles,
    base_path="scenarios/double_integrator/base.xml",
    worldbody_path="scenarios/double_integrator/world.xml",
    assets_path="scenarios/double_integrator/assets.xml",
    defaults_path="scenarios/double_integrator/defaults.xml"
)

# Robot actuation
actuator_force_x = bindings["robot1"]["actuators"]["force_x"]
actuator_force_y = bindings["robot1"]["actuators"]["force_y"]
actuator_force_z = bindings["robot1"]["actuators"]["force_z"]

DT = m.opt.timestep

# Hide UI by default
with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:

    pb = Playback()
    step = 0

    # Initial robot state
    x0, y0, z0 = get_3d_position(d, bindings["robot1"]["bodies"]["robot"])
    vx0, vy0, vz0 = get_3d_velocity(d, bindings["robot1"]["bodies"]["robot"])

    controller = QP3D(
        dt=DT,
        target=target,
        initial_state=np.array([x0, y0, z0, vx0, vy0, vz0 ]),
        sh_n=sh_n,
        sh_tau=sh_tau,
        obstacles=get_collision_spheres(['robot'])
    )

    controller.set_max_accel(max_accel)
    controller.set_reference_speed(ref_speed)

    # generate_new_target()
    controller.set_target(target)

    while viewer.is_running():
        step_start = time.time()

        # Playback control:
        #           space to start/freeze sim
        if pb.step > 0:
            pb.step = pb.step - 1 
        elif pb.paused:
            time.sleep(0.1)
            continue

        # Robot state
        x, y, z = get_3d_position(d, bindings["robot1"]["bodies"]["robot"])
        vx, vy, vz = get_3d_velocity(d, bindings["robot1"]["bodies"]["robot"])

        x_vec = np.array(( x, y, z, vx, vy, vz))

        controller.update_state(x_vec)
        
        controller.update_obstacles(get_collision_spheres(['robot']))

        # ------------------------------------------
        # Control
        # ------------------------------------------

        a_cmd = controller.compute_command()
        
        d.ctrl[actuator_force_x] = a_cmd[0]
        d.ctrl[actuator_force_y] = a_cmd[1]
        d.ctrl[actuator_force_z] = a_cmd[2]

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        dist = np.linalg.norm(target - x_vec[:3])

        # ------------------------------------------
        # Visualization
        # ------------------------------------------

        with viewer.lock():

            scene = viewer.user_scn
            scene.ngeom = 0


            ratio = np.linalg.norm(a_cmd) / 0.15 

            # acceleration command
            draw_vector(
                scene,
                x_vec[:3] + a_cmd / ratio,
                a_cmd * 2,
                [1, 1, 0, 0.8]
            )

            draw_sphere(
                scene,
                target,
                (0, 1, 0, 1),
                0.06
            )

            if pb.show_obstacles_collision_boxes:

                for obstacle_name, obstacle in get_collision_spheres(['robot']).items():
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

        # log.append([
        #     d.time,
        #     x,
        #     y,
        #     yaw,
        #     v_cmd,
        #     w_cmd,
        #     dist,
        #     yaw_err
        # ])

        if step % 80 == 0:
            print(f"==============================")
            print(
                f"[t={d.time:6.3f}s] "
                f"state={' '.join(f'{x:6.3f}' for x in x_vec)} "
                f"dist={dist:6.4f} "
                f"u={' '.join(f'{a:6.4f}' for a in a_cmd)}"
            )
            print(
                format_obstacles(get_collision_spheres())
            )

        if True:
            if abs(dist) <= 0.005 and np.linalg.norm(x_vec[3:]) <= 0.005:
                generate_new_target()

        step += 1

        time_until_next_step = DT - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

controller.stop()
