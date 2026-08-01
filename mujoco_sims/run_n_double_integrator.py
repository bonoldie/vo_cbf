import time
import csv
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
from pathlib import Path
import cv2 as cv2
import os
import jax

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
from controllers.qp_3d_profiled import QP3DProfiled

# SH params
sh_n = 6
sh_tau = 1.2

# Control params
ref_speed = 0.2
max_accel = 2.0

collision_radius = 0.15
controllers = []

# Scenario
obstacles = [] 

r = 1
n = 2

robots = [
    {
        "name": "robot" + str(i),
        "collision_radius": collision_radius,
        "pos": (r * np.cos((2* np.pi / n) * i), r * np.sin((2* np.pi / n) * i), 0),
        "robot_path": "scenarios/double_integrator/bot.xml"
    } for i in range(n)
]

targets =  dict(zip(
    [robot["name"] for robot in robots],
    [np.asarray((r * np.cos((2* np.pi / n) * i + (np.pi)), r * np.sin((2* np.pi / n) * i + (np.pi)), (1 if np.random.rand() >= 0.5 else -1) * np.random.rand()/20 )) for i in range(n)]
    )
)

m, d, bindings, get_collision_spheres = buildModel(
    robots,
    obstacles,
    base_path="scenarios/double_integrator/base.xml",
    worldbody_path="scenarios/double_integrator/world.xml",
    assets_path="scenarios/double_integrator/assets.xml",
    defaults_path="scenarios/double_integrator/defaults.xml"
)

# Robot actuation

robots_actuators = {}

for robot in robots:
    robots_actuators[robot["name"]] = (
        bindings[robot["name"]]["actuators"]["force_x"],
        bindings[robot["name"]]["actuators"]["force_y"],
        bindings[robot["name"]]["actuators"]["force_z"]
    )

DT = m.opt.timestep

VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_PATH = Path("videos/n_double_integrator_sim.mp4")

m.vis.global_.offwidth = VIDEO_WIDTH
m.vis.global_.offheight = VIDEO_HEIGHT

renderer = mujoco.Renderer(
    m,
    width=VIDEO_WIDTH,
    height=VIDEO_HEIGHT,
    max_geom=10_000,
)

# camera setup
CAMERA_DISTANCE = 5.71095
CAMERA_AZIMUTH = 68.75
CAMERA_ELEVATION = -49.75
CAMERA_LOOKAT = np.array([0.39250632,  1.30607708, -2.06329618])

render_camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(render_camera)

render_camera.distance = CAMERA_DISTANCE
render_camera.azimuth = CAMERA_AZIMUTH
render_camera.elevation = CAMERA_ELEVATION
render_camera.lookat[:] = CAMERA_LOOKAT

next_frame_time = float(d.time)
frame_period = 1.0 / VIDEO_FPS
frames = []

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (VIDEO_WIDTH,VIDEO_HEIGHT))

# ======================================================================
# Custom visualization
# ======================================================================

def draw_custom_geometries(
    scene,
    robots_current_state,
    show_collision_spheres,
):
    # print(robots_current_state)
    for robot_name, robot_state in robots_current_state.items():
        robot_position = robot_state['position']
        robot_velocity = robot_state['velocity']
        robot_command = robot_state['command']
        
        command_norm = np.linalg.norm(robot_command)
        speed = np.linalg.norm(robot_velocity)


    
        # Acceleration command arrow.
        if command_norm > 1e-4:
            arrow_start = robot_position + (
                robot_command / command_norm
            ) * collision_radius

            draw_vector(    
                scene,
                arrow_start,
                robot_command,
                [1.0, 1.0, 0.0, 0.8],
            )

        if speed > 1e-4:
            velocity_arrow_start = robot_position + (
                robot_velocity / speed
            ) * collision_radius

            draw_vector(    
                scene,
                velocity_arrow_start,
                robot_velocity,
                [0.0, 1.0, 0.0, 0.8],
            )


    for robot_name, target in targets.items():
        # Target.
        draw_sphere(
            scene,
            np.asarray(target),
            (0.0, 1.0, 0.0, 1.0),
            0.06,
        )

    # Collision spheres.
    if show_collision_spheres:
        obstacles = get_collision_spheres()
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

        # pb = Playback()
        step = 0
        real_start_time = time.time()


       
        controllers = {}

        for robot in robots:
            x0, y0, z0 = get_3d_position(
                d,
                bindings[robot["name"]]["bodies"]["robot"],
            )

            vx0, vy0, vz0 = get_3d_velocity(
                d,
                bindings[robot["name"]]["bodies"]["robot"],
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

            # controller = QP3D(
            #     dt=DT,
            #     target=targets[robot["name"]],
            #     initial_state=initial_state,
            #     collision_radius=collision_radius,
            #     sh_n=sh_n,
            #     sh_tau=sh_tau,
            #     obstacles=get_collision_spheres(robot["name"]),
            #     device_id=0
            # )

            controller = QP3DProfiled(
                dt=DT,
                target=targets[robot["name"]],
                initial_state=initial_state,
                collision_radius=collision_radius,
                obstacles=get_collision_spheres(robot["name"]),

                # Index in jax.devices("gpu")
                device_id=0,

                profile=True,
                profile_every=1,
            )

            controller.set_max_accel(max_accel)
            controller.set_reference_speed(ref_speed)

            controllers[robot["name"]] = controller

        # --------------------------------------------------------------
        # Main loop
        # --------------------------------------------------------------

        while viewer.is_running():
            step_start = time.time()

            # print(viewer.cam)
            # ----------------------------------------------------------
            # Playback control
            # ----------------------------------------------------------

            # if pb.step > 0:
            #     pb.step -= 1
# 
            # elif pb.paused:
            #     viewer.sync()
            #     time.sleep(0.05)
            #     continue

            # ----------------------------------------------------------
            # Read robot state
            # ----------------------------------------------------------
            state = {}

            for robot in robots:
                state[robot["name"]] = {}

                x, y, z = get_3d_position(
                    d,
                    bindings[robot["name"]]["bodies"]["robot"],
                )

    
                vx, vy, vz = get_3d_velocity(
                    d,
                    bindings[robot["name"]]["bodies"]["robot"],
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

                state[robot["name"]]["position"] = robot_state[:3]
                state[robot["name"]]["velocity"] = robot_state[3:]
    
                # ----------------------------------------------------------
                # Update controller
                # ----------------------------------------------------------

                controller = controllers[robot["name"]]

                controller.update_state(robot_state)
    
                controller.update_obstacles(
                    get_collision_spheres([robot["name"]])
                )
    
                acceleration_command = np.asarray(
                    controller.compute_command(),
                    dtype=float,
                )

                state[robot["name"]]["command"] = acceleration_command
        
                # ----------------------------------------------------------
                # Apply control
                # ----------------------------------------------------------
    
                d.ctrl[robots_actuators[robot["name"]][0]] = acceleration_command[0]
                d.ctrl[robots_actuators[robot["name"]][1]] = acceleration_command[1]
                d.ctrl[robots_actuators[robot["name"]][2]] = acceleration_command[2]

                controller.increment_step()
          

            # ----------------------------------------------------------
            # Interactive viewer visualization
            # ----------------------------------------------------------

            with viewer.lock():
                viewer.user_scn.ngeom = 0

                draw_custom_geometries(
                    scene=viewer.user_scn,
                    robots_current_state=state,
                    show_collision_spheres=False, # pb.show_obstacles_collision_boxes
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

                print("FRAME SAVED!!!!")
                # This adds the regular MuJoCo model to renderer.scene.
                renderer.update_scene(
                    d,
                    camera=render_camera,
                )

                # Add target, acceleration arrow and collision spheres.
                draw_custom_geometries(
                    scene = renderer.scene,
                    robots_current_state=state,
                    show_collision_spheres=False # pb.show_obstacles_collision_boxes
                )

                frame = renderer.render()
                video_out.write(frame[:, :, ::-1])

                next_frame_time += frame_period

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------

            if step % 80 == 0:
                print("==============================")

                print(
                    f"[t={d.time:6.3f}s ({d.time / (time.time() - real_start_time):6.3f}x)] "
                    # f"state={format_vector(robot_state)} "
                    # f"u={format_vector(acceleration_command)}"
                )

                print(
                    format_obstacles(
                        get_collision_spheres()
                    )
                )

            # ----------------------------------------------------------
            # Generate a new target
            # ----------------------------------------------------------

            # velocity_norm = np.linalg.norm(robot_state[3:])
            # if (
            #     distance_to_target <= 0.005
            #     and velocity_norm <= 0.005
            # ):
            #     generate_new_target()

            step += 1
            controller.increment_step()

            # ----------------------------------------------------------
            # Optional real-time synchronization
            # ----------------------------------------------------------

            # remaining_time = DT - (time.time() - step_start)
            #
            # if remaining_time > 0.0:
            #     time.sleep(remaining_time)
except:
    print("Error")
    
finally:
    # if controller is not None:
    #     controller.stop()

    renderer.close()


# ======================================================================
# Save and display video
# ======================================================================

def append_raw(filename):
  return "{0}_{2}{1}".format(*os.path.splitext(filename) + ('x264',))

if True:
    video_out.release()

    os.system(f"ffmpeg -i {VIDEO_PATH} -y -vcodec libx264 {append_raw(VIDEO_PATH)} > /dev/null 2>&1")

    print(f"Video saved to: {VIDEO_PATH.resolve()}")

