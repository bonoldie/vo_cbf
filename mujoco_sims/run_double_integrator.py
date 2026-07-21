import time
import csv
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
from pathlib import Path

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
target = np.array([4.0, 4.0, 3.8])

# SH params
sh_n = 6
sh_tau = 1.2

# Control params
ref_speed = 0.2
max_accel = 2.0

collision_radius = 0.15
controller = None

target_side = 1

def generate_new_target(margin=1.0):
    global target, target_side, controller, obstacles

    obstacle_positions = np.asarray(
        [obstacle["pos"] for obstacle in obstacles],
        dtype=float,
    )

    cloud_min = obstacle_positions.min(axis=0)
    cloud_max = obstacle_positions.max(axis=0)

    if target_side > 0:
        target_x = cloud_max[0] + margin
    else:
        target_x = cloud_min[0] - margin

    target_side *= -1

    target = np.array(
        [
            target_x,
            np.random.uniform(cloud_min[1], cloud_max[1]),
            np.random.uniform(
                max(0.2, cloud_min[2]),
                cloud_max[2],
            ),
        ],
        dtype=float,
    )

    controller.set_target(target)

    return target

# def generate_new_target():
#     global target, controller
#     target = np.array([np.random.uniform(-2.0,2.0), np.random.uniform(-2.0,2.0),np.random.uniform(0.2,4.0)])
#     controller.set_target(target)

# obstacles = [
#     {
#         "type": ObstacleType.SPHERE,
#         "pos": (1, 1, 1),
#         "radius": 0.25,
#         
#     },
#     {
#         "type": ObstacleType.SPHERE,
#         "pos": (2, 2, 2),
#         "radius": 0.25
#     }
# ]

def generate_obstacles(
    grid_size=(2, 3),
    density=2,
    cell_size=1.0,
    radius=0.25,
    z_range=(0.5, 5.0),
    seed=None,
):
    """
    Generate `density` spherical obstacles inside each grid cell.

    grid_size:
        Number of cells along x and y.

    density:
        Number of obstacles generated per cell.

    cell_size:
        Width and height of each grid cell.
    """
    rng = np.random.default_rng(seed)
    obstacles = []

    rows, columns = grid_size

    for i in range(rows):
        for j in range(columns):
            for _ in range(density):
                # Random position inside the current grid cell
                x = rng.uniform(i * cell_size, (i + 1) * cell_size)
                y = rng.uniform(j * cell_size, (j + 1) * cell_size)
                z = rng.uniform(*z_range)

                obstacles.append(
                    {
                        "type": ObstacleType.SPHERE,
                        "pos": (float(x), float(y), float(z)),
                        "radius": radius,
                    }
                )

    return obstacles


obstacles = generate_obstacles(
    grid_size=(1, 1),
    density=1,
    cell_size=3.0,
    z_range=(0.5, 5.0),
    seed=43,
)

# obstacles = []

m, d, bindings, get_collision_spheres = buildModel(
    [
        {
            "name": "robot1",
            "collision_radius": collision_radius,
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


VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_PATH = Path("mujoco_simulation.mp4")

m.vis.global_.offwidth = VIDEO_WIDTH
m.vis.global_.offheight = VIDEO_HEIGHT

renderer = mujoco.Renderer(
    m,
    width=VIDEO_WIDTH,
    height=VIDEO_HEIGHT,
    max_geom=10_000,
)

# camera setup
CAMERA_DISTANCE = 10.12
CAMERA_AZIMUTH = 118.5
CAMERA_ELEVATION = -24.0
CAMERA_LOOKAT = np.array([2.0, 2.0, 2.0])

render_camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(render_camera)

render_camera.distance = CAMERA_DISTANCE
render_camera.azimuth = CAMERA_AZIMUTH
render_camera.elevation = CAMERA_ELEVATION
render_camera.lookat[:] = CAMERA_LOOKAT

next_frame_time = float(d.time)
frame_period = 1.0 / VIDEO_FPS
frames = []

# ======================================================================
# Custom visualization
# ======================================================================

def draw_custom_geometries(
    scene,
    robot_state,
    acceleration_command,
    current_target,
    show_collision_spheres,
):
    position = robot_state[:3]
    acceleration_norm = np.linalg.norm(acceleration_command)

    # Acceleration command arrow.
    if acceleration_norm > 1e-9:
        arrow_length = 0.15
        arrow_start = position + (
            acceleration_command / acceleration_norm
        ) * arrow_length

        draw_vector(
            scene,
            arrow_start,
            acceleration_command * 2.0,
            [1.0, 1.0, 0.0, 0.8],
        )

    # Target.
    draw_sphere(
        scene,
        np.asarray(current_target),
        (0.0, 1.0, 0.0, 1.0),
        0.06,
    )

    # Collision spheres.
    if show_collision_spheres:
        obstacles = get_collision_spheres(["robot"])

        for obstacle in obstacles.values():
            draw_sphere(
                scene,
                np.asarray(obstacle["p"]),
                (0.0, 0.0, 1.0, 0.1),
                obstacle["collision_radius"],
            )

def format_vector(vector):
    return " ".join(
        f"{value:6.3f}"
        for value in vector
    )

# Hide UI by default
# with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:

#     # camera setup
#     viewer.cam.distance = 10.12
#     viewer.cam.azimuth = 118.5
#     viewer.cam.elevation = -24.0
#     viewer.cam.lookat[:] = np.array([2.0, 2.0, 2.0])

#     # sim setup
#     pb = Playback()
#     step = 0

#     # Initial robot state
#     x0, y0, z0 = get_3d_position(d, bindings["robot1"]["bodies"]["robot"])
#     vx0, vy0, vz0 = get_3d_velocity(d, bindings["robot1"]["bodies"]["robot"])

#     controller = QP3D(
#         dt=DT,
#         target=target,
#         initial_state=np.array([x0, y0, z0, vx0, vy0, vz0 ]),
#         collision_radius=collision_radius,
#         sh_n=sh_n,
#         sh_tau=sh_tau,
#         obstacles=get_collision_spheres(['robot'])
#     )

#     controller.set_max_accel(max_accel)
#     controller.set_reference_speed(ref_speed)

#     # generate_new_target()
#     controller.set_target(target)

#     while viewer.is_running():
#         step_start = time.time()

#         # Playback control:
#         #           space to start/freeze sim
#         if pb.step > 0:
#             pb.step = pb.step - 1 
#         elif pb.paused:
#             time.sleep(0.1)
#             continue

#         # Robot state
#         x, y, z = get_3d_position(d, bindings["robot1"]["bodies"]["robot"])
#         vx, vy, vz = get_3d_velocity(d, bindings["robot1"]["bodies"]["robot"])

#         x_vec = np.array(( x, y, z, vx, vy, vz))

#         controller.update_state(x_vec)
        
#         controller.update_obstacles(get_collision_spheres(['robot1']))

#         # ------------------------------------------
#         # Control
#         # ------------------------------------------

#         a_cmd = controller.compute_command()
        
#         d.ctrl[actuator_force_x] = a_cmd[0]
#         d.ctrl[actuator_force_y] = a_cmd[1]
#         d.ctrl[actuator_force_z] = a_cmd[2]

#         # ------------------------------------------
#         # Metrics
#         # ------------------------------------------

#         dist = np.linalg.norm(target - x_vec[:3])

#         # ------------------------------------------
#         # Visualization
#         # ------------------------------------------

#         with viewer.lock():

#             scene = viewer.user_scn
#             scene.ngeom = 0

#             ratio = np.linalg.norm(a_cmd) / 0.15 

#             # acceleration command
#             draw_vector(
#                 scene,
#                 x_vec[:3] + a_cmd / ratio,
#                 a_cmd * 2,
#                 [1, 1, 0, 0.8]
#             )

#             draw_sphere(
#                 scene,
#                 target,
#                 (0, 1, 0, 1),
#                 0.06
#             )

#             if pb.show_obstacles_collision_boxes:

#                 for obstacle_name, obstacle in get_collision_spheres(['robot']).items():
#                     draw_sphere(
#                         scene,
#                         np.array(obstacle["p"]),
#                         (0, 0, 1, 0.1),
#                         obstacle["collision_radius"]
#                     )


#         mujoco.mj_step(m, d)
#         viewer.sync()

#         if len(frames) < d.time * framerate:
#             print(viewer)
#             #pixels = viewer.render()
#             #frames.append(pixels)

#         # ------------------------------------------
#         # Logging
#         # ------------------------------------------

#         # log.append([
#         #     d.time,
#         #     x,
#         #     y,
#         #     yaw,
#         #     v_cmd,
#         #     w_cmd,
#         #     dist,
#         #     yaw_err
#         # ])

#         if step % 80 == 0:
#             print(f"==============================")
#             print(
#                 f"[t={d.time:6.3f}s] "
#                 f"state={' '.join(f'{x:6.3f}' for x in x_vec)} "
#                 f"dist={dist:6.4f} "
#                 f"u={' '.join(f'{a:6.4f}' for a in a_cmd)}"
#             )
#             print(
#                 format_obstacles(get_collision_spheres())
#             )

#         if True:
#             if abs(dist) <= 0.005 and np.linalg.norm(x_vec[3:]) <= 0.005:
#                 generate_new_target()

#         step += 1
#         controller.increment_step()

#         # time_until_next_step = DT - (time.time() - step_start)
#         # if time_until_next_step > 0:
#         #     time.sleep(time_until_next_step)

# controller.stop()

try:
    with mujoco.viewer.launch_passive(
        m,
        d,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:

        # --------------------------------------------------------------
        # Interactive viewer camera
        # --------------------------------------------------------------

        viewer.cam.distance = CAMERA_DISTANCE
        viewer.cam.azimuth = CAMERA_AZIMUTH
        viewer.cam.elevation = CAMERA_ELEVATION
        viewer.cam.lookat[:] = CAMERA_LOOKAT

        # --------------------------------------------------------------
        # Simulation setup
        # --------------------------------------------------------------

        pb = Playback()
        step = 0

        x0, y0, z0 = get_3d_position(
            d,
            bindings["robot1"]["bodies"]["robot"],
        )

        vx0, vy0, vz0 = get_3d_velocity(
            d,
            bindings["robot1"]["bodies"]["robot"],
        )

        initial_state = np.array(
            [
                x0,
                y0,
                z0,
                vx0,
                vy0,
                vz0,
            ],
            dtype=float,
        )

        controller = QP3D(
            dt=DT,
            target=target,
            initial_state=initial_state,
            collision_radius=collision_radius,
            sh_n=sh_n,
            sh_tau=sh_tau,
            obstacles=get_collision_spheres(["robot"]),
        )

        controller.set_max_accel(max_accel)
        controller.set_reference_speed(ref_speed)
        controller.set_target(target)

        # --------------------------------------------------------------
        # Main loop
        # --------------------------------------------------------------

        while viewer.is_running():
            step_start = time.time()

            # ----------------------------------------------------------
            # Playback control
            # ----------------------------------------------------------

            if pb.step > 0:
                pb.step -= 1

            elif pb.paused:
                viewer.sync()
                time.sleep(0.05)
                continue

            # ----------------------------------------------------------
            # Read robot state
            # ----------------------------------------------------------

            x, y, z = get_3d_position(
                d,
                bindings["robot1"]["bodies"]["robot"],
            )

            vx, vy, vz = get_3d_velocity(
                d,
                bindings["robot1"]["bodies"]["robot"],
            )

            robot_state = np.array(
                [
                    x,
                    y,
                    z,
                    vx,
                    vy,
                    vz,
                ],
                dtype=float,
            )

            # ----------------------------------------------------------
            # Update controller
            # ----------------------------------------------------------

            controller.update_state(robot_state)

            controller.update_obstacles(
                get_collision_spheres(["robot1"])
            )

            acceleration_command = np.asarray(
                controller.compute_command(),
                dtype=float,
            )

            # ----------------------------------------------------------
            # Apply control
            # ----------------------------------------------------------

            d.ctrl[actuator_force_x] = acceleration_command[0]
            d.ctrl[actuator_force_y] = acceleration_command[1]
            d.ctrl[actuator_force_z] = acceleration_command[2]

            distance_to_target = np.linalg.norm(
                target - robot_state[:3]
            )

            # ----------------------------------------------------------
            # Interactive viewer visualization
            # ----------------------------------------------------------

            with viewer.lock():
                viewer.user_scn.ngeom = 0

                draw_custom_geometries(
                    scene=viewer.user_scn,
                    robot_state=robot_state,
                    acceleration_command=acceleration_command,
                    current_target=target,
                    show_collision_spheres=(
                        pb.show_obstacles_collision_boxes
                    ),
                )

            # ----------------------------------------------------------
            # Advance simulation
            # ----------------------------------------------------------

            mujoco.mj_step(m, d)
            viewer.sync()

            # ----------------------------------------------------------
            # Record video frame
            # ----------------------------------------------------------

            if d.time >= next_frame_time:
                # This adds the regular MuJoCo model to renderer.scene.
                renderer.update_scene(
                    d,
                    camera=render_camera,
                )

                # Add target, acceleration arrow and collision spheres.
                draw_custom_geometries(
                    scene=renderer.scene,
                    robot_state=robot_state,
                    acceleration_command=acceleration_command,
                    current_target=target,
                    show_collision_spheres=(
                        pb.show_obstacles_collision_boxes
                    ),
                )

                frame = renderer.render()
                frames.append(frame.copy())

                next_frame_time += frame_period

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------

            if step % 80 == 0:
                print("==============================")

                print(
                    f"[t={d.time:6.3f}s] "
                    f"state={format_vector(robot_state)} "
                    f"dist={distance_to_target:6.4f} "
                    f"u={format_vector(acceleration_command)}"
                )

                print(
                    format_obstacles(
                        get_collision_spheres()
                    )
                )

            # ----------------------------------------------------------
            # Generate a new target
            # ----------------------------------------------------------

            velocity_norm = np.linalg.norm(robot_state[3:])

            if (
                distance_to_target <= 0.005
                and velocity_norm <= 0.005
            ):
                generate_new_target()

            step += 1
            controller.increment_step()

            # ----------------------------------------------------------
            # Optional real-time synchronization
            # ----------------------------------------------------------

            # remaining_time = DT - (time.time() - step_start)
            #
            # if remaining_time > 0.0:
            #     time.sleep(remaining_time)

finally:
    # if controller is not None:
    #     controller.stop()

    renderer.close()


# ======================================================================
# Save and display video
# ======================================================================

if not frames:
    print("No frames were recorded.")

else:
    duration = len(frames) / VIDEO_FPS

    print(
        f"Recorded {len(frames)} frames "
        f"({duration:.2f} seconds)."
    )

    media.write_video(
        VIDEO_PATH,
        frames,
        fps=VIDEO_FPS,
    )

    print(f"Video saved to: {VIDEO_PATH.resolve()}")

    # Displays the video in Jupyter/IPython.
    media.show_video(
        frames,
        fps=VIDEO_FPS,
    )

