import time
import csv
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
from pathlib import Path
import cv2 as cv2
import os

from generate_scenarios import buildModel, format_obstacles
from utils.playback import Playback
from utils.scenebuilder import ObstacleType
from utils.utils import (
    draw_sphere,
    draw_vector,
    get_3d_position,
    get_3d_site_position,
    get_3d_velocity,
    get_3d_orientation,
    get_3d_angular_velocity,
    get_mass,
    get_inertia
)

from controllers.qp_3d_precomp_drone import QP3DPrecompDrone

# --------------------------------------------------
# Scenario
# --------------------------------------------------
target = np.array([0.0, 0.0, 1.0])

# SH params
sh_n = 6
sh_tau = 1.2

# Control params
ref_speed = 0.2
max_accel = 2.0

collision_radius = 0.15
controller = None

target_side = 1

def generate_new_target(margin=0.5):
    global target, target_heading, target_side, controller, obstacles

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

    yaw = np.random.uniform(-np.pi, np.pi)
    target_heading = np.asarray((np.cos(yaw),np.sin(yaw), 0.0))

    # controller.set_target(target)

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

def get_drone_state(d, robot_body_id):
    x0, y0, z0 = get_3d_position(
        d,
        robot_body_id,
    )

    R0 = get_3d_orientation(
        d,
        robot_body_id,
    )

    vx0, vy0, vz0 = get_3d_velocity(
        d,
        robot_body_id,
    )

    omegax0, omegay0, omegaz0 = get_3d_angular_velocity(
            d,
            robot_body_id,
        )

    return np.concatenate((
        np.array([x0, y0, z0], dtype=float),
        np.array([vx0, vy0, vz0], dtype=float),
        np.asarray(R0, dtype=float).reshape(-1),
        np.array([omegax0, omegay0, omegaz0], dtype=float),
    ))


obstacles = generate_obstacles(
    grid_size=(2, 2),
    density=2,
    cell_size=1,
    z_range=(0.5, 1.5),
    seed=44,
)

obstacles = []

m, d, bindings, get_collision_spheres = buildModel(
    [
        {
            "name": "robot1",
            "collision_radius": collision_radius,
            "pos": (0, 0, 0),
            "robot_path": "scenarios/drone/skydio_x2.xml"
        }
        # ,
        # {
        #             "name": "robot2",
        #             "collision_radius": collision_radius,
        #             "pos": (0, 0, 2),
        #             "robot_path": "scenarios/drone/skydio_x2.xml"
        #         }
    ],
    obstacles,
    base_path="scenarios/drone/base.xml",
    worldbody_path="scenarios/drone/world.xml",
    assets_path="scenarios/drone/assets.xml",
    defaults_path="scenarios/drone/defaults.xml"
)


# Robot actuation
actuator_thrust1 = bindings["robot1"]["actuators"]["thrust1"]
actuator_thrust2 = bindings["robot1"]["actuators"]["thrust2"]
actuator_thrust3 = bindings["robot1"]["actuators"]["thrust3"]
actuator_thrust4 = bindings["robot1"]["actuators"]["thrust4"]

mass = get_mass(m, bindings["robot1"]["bodies"]["skydio_x2"])
inertia = get_inertia(m, bindings["robot1"]["bodies"]["skydio_x2"])
# rotor distance
D = 0.23345235059857505

thrusts_to_T_M = np.array((
    ( 1, 1,  1, 1 ),
    ( 0, -D, 0, D ),
    ( D, 0, -D, 0 ),
    ( -1, 1,-1, 1 ))
,dtype=float)

T_M_to_thrusts = np.linalg.pinv(thrusts_to_T_M)

def map_T_M_to_thrusts(T: float, M: np.ndarray):
    return  T_M_to_thrusts @ np.concatenate((np.asarray([T], dtype=float), M))


def map_thrusts_to_T_M(thrusts: np.ndarray):
    T_M = thrusts_to_T_M @ thrusts
    
    return T_M[0], T_M[1:]
     


DT = m.opt.timestep

VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_PATH = Path("videos/drone_sim.mp4")

m.vis.global_.offwidth = VIDEO_WIDTH
m.vis.global_.offheight = VIDEO_HEIGHT

renderer = mujoco.Renderer(
    m,
    width=VIDEO_WIDTH,
    height=VIDEO_HEIGHT,
    max_geom=10_000,
)

# camera setup
CAMERA_DISTANCE = 4.459003270
CAMERA_AZIMUTH = 95.25
CAMERA_ELEVATION = -42.5
CAMERA_LOOKAT = np.array([0.56715552, 1.22202891, 0.1062733])

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
    robot_state,
    thrust_commands,
    show_collision_spheres,
):
    position = robot_state[:3]
    drone_orientation = robot_state[6:15].reshape(3,3)
    thrusts_norm = np.linalg.norm(thrust_commands)
    speed = np.linalg.norm(robot_state[3:6])
    arrow_length = 0.15

    # Acceleration command arrow.
    if thrusts_norm > 1e-9:

        for i in range(4):
            motor_position = np.asarray(get_3d_site_position(d, bindings["robot1"]["sites"]["".join(["thrust", str(i+1)])]), dtype=float)

            # print(motor_position)
            # print(drone_orientation[:, 2] * thrust_commands[i] * arrow_length)

            draw_vector(    
                scene,
                motor_position,
                drone_orientation[:, 2] * thrust_commands[i] * arrow_length,
                [1.0, 1.0, 0.0, 0.8],
            )

    draw_vector(    
        scene,
        position,
        robot_state[12:15] * arrow_length,
        [0.0, 1.0, 0.0, 0.8],
    )


    # Target
    draw_sphere(
        scene,
        np.asarray(target),
        (0.0, 1.0, 0.0, 1.0),
        0.06,
    )

    #draw_vector(
    #    scene,
    #    np.asarray(target),
    #    target_heading * arrow_length * 2,
    #    [0.2, 1.0, 0.2, 0.8],
    # )

    # Collision spheres.
    if show_collision_spheres:
        obstacles = get_collision_spheres(["skydio_x2"], robot_body_name = "skydio_x2")

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
        show_left_ui=True,
        show_right_ui=True,
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
        real_start_time = time.time()

        initial_state = get_drone_state(d, bindings["robot1"]["bodies"]["skydio_x2"])

        controller = QP3DPrecompDrone(
            dt=DT,
            mass=mass,
            inertia=np.diag(inertia),
            target=target,
            initial_state=initial_state,
            collision_radius=collision_radius,
            sh_n=sh_n,
            sh_tau=sh_tau,
            obstacles=get_collision_spheres(["robot1"], robot_body_name="skydio_x2")
        )

        # controller.set_max_accel(max_accel)
        # controller.set_reference_speed(ref_speed)
        # generate_new_target() # controller.set_target(target)

        # --------------------------------------------------------------
        # Main loop
        # --------------------------------------------------------------

        while viewer.is_running():
            step_start = time.time()

            # print(viewer.cam)
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

            robot_state = get_drone_state(d, bindings["robot1"]["bodies"]["skydio_x2"])

            # ----------------------------------------------------------
            # Update controller
            # ----------------------------------------------------------

            controller.update_state(robot_state)

            controller.update_obstacles(
                get_collision_spheres(["robot1"], robot_body_name="skydio_x2")
            )

            # T is the total thrust
            # M R(3x1) are the moments 
            T, M, obstacles_states =  controller.compute_command()
            
            
            control, boundaries, reference_data = controller.compute_command()

            thrust_command = control[0]
            moment_command = np.asarray((control[2], control[1], -control[3]), dtype=float) 

            thrust_commands = map_T_M_to_thrusts(thrust_command, moment_command)

            # thrust_commands = np.array([2.0,2.0,2.0,2.0]) # np.zeros(4) 

            # ----------------------------------------------------------
            # Apply control
            # ----------------------------------------------------------
            
            thrust_commands = np.array([4.0, 3.0, 2.0, 1.0])
            
            d.ctrl[actuator_thrust1] = thrust_commands[0]
            d.ctrl[actuator_thrust2] = thrust_commands[3]
            d.ctrl[actuator_thrust3] = thrust_commands[2]
            d.ctrl[actuator_thrust4] = thrust_commands[1]

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
                    thrust_commands=thrust_commands,
                    show_collision_spheres= ()
                    # show_collision_spheres=(
                    #     pb.show_obstacles_collision_boxes
                    # ),
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
                    camera=render_camera
                )

                # Add target, acceleration arrow and collision spheres.
                draw_custom_geometries(
                    scene=viewer.user_scn,
                    robot_state=robot_state,
                    thrust_commands=thrust_commands,
                    show_collision_spheres=()
                    # show_collision_spheres=(
                    #     pb.show_obstacles_collision_boxes
                    # )
                )

                frame = renderer.render()
                # frames.append(frame.copy())

                video_out.write(frame[:, :, ::-1])

                next_frame_time += frame_period

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------

            if step % 80 == 0:
                print("==============================")
                print(
                    f"[t={d.time:6.3f}s ({d.time / (time.time() - real_start_time):6.3f}x)] "
                    f"state={format_vector(robot_state)} "
                    f"dist={distance_to_target:6.4f} "
                    f"u={format_vector(thrust_commands)}"
                )

                print(
                    format_obstacles(
                        get_collision_spheres(robot_body_name = "skydio_x2")
                    )
                )

            # ----------------------------------------------------------
            # Generate a new target
            # ----------------------------------------------------------

            velocity_norm = np.linalg.norm(robot_state[12:15])

            if (
                distance_to_target <= 0.005
                and velocity_norm <= 0.005
            ):
                generate_new_target()

            step += 1
            # controller.increment_step()

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

def append_raw(filename):
  return "{0}_{2}{1}".format(*os.path.splitext(filename) + ('x264',))
if True:
    duration = len(frames) / VIDEO_FPS

    video_out.release()
    os.system(f"ffmpeg -i {VIDEO_PATH} -y -vcodec libx264 {append_raw(VIDEO_PATH)} > /dev/null 2>&1")

    print(f"Video saved to: {VIDEO_PATH.resolve()}")

