
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
from utils.utils import *
from control_world_plot import ControlWorldPlot
from controllers.qp_3d_precomp_drone import QP3DPrecompDrone

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


obstacles = []

m, d, bindings, get_collision_spheres = buildModel(
    [
        {
            "name": "robot1",
            "collision_radius": 0.3,
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

# Actuation
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


R1 = rotation_matrix('x', -np.pi)
R2 = rotation_matrix(R1[:, 2], -np.pi/2)

sim_world_to_control_world = R1 @ R2

align_rotors = rotation_matrix(sim_world_to_control_world[:, 2], (3 * np.pi)/4)
     
sim_world_to_control_world_with_aligned_rotors =  sim_world_to_control_world @ align_rotors

DT = m.opt.timestep

control_world_plot = ControlWorldPlot(
    sim_world_to_control_world_with_aligned_rotors,
    axis_length=0.5,
    plot_limit=3.0,
)

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

       
        pb = Playback()
        step = 0
        real_start_time = time.time()

        initial_state = get_drone_state(d, bindings["robot1"]["bodies"]["skydio_x2"])

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

            thrust_command = 0.0
            moment_command = np.asarray((0.0, 0.0, 0.0), dtype=float) 

            thrust_commands = map_T_M_to_thrusts(thrust_command, moment_command)
            
            # ----------------------------------------------------------
            # Apply control
            # ----------------------------------------------------------
            
            thrust_commands = np.array([4.0, 3.0, 2.0, 1.0])
            
            d.ctrl[actuator_thrust1] = thrust_commands[0]
            d.ctrl[actuator_thrust2] = thrust_commands[3]
            d.ctrl[actuator_thrust3] = thrust_commands[2]
            d.ctrl[actuator_thrust4] = thrust_commands[1]
            
            # ----------------------------------------------------------
            # Interactive viewer visualization
            # ----------------------------------------------------------

            if step % 5 == 0:
                control_world_plot.update(robot_state)
                
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
            # Optional real-time synchronization
            # ----------------------------------------------------------

            remaining_time = DT - (time.time() - step_start)
            if remaining_time > 0.0:
                time.sleep(remaining_time)

except:
    print("ERROR")