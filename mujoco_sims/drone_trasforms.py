import time
import csv
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import cv2 as cv2
import os

from generate_scenarios import buildModel, format_obstacles
from utils.playback import Playback
from utils.scenebuilder import ObstacleType
from utils.utils import *
from control_world_plot import ControlWorldPlot
from controllers.qp_3d_precomp_drone import QP3DPrecompDrone

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
torque_constant = 0.0201

# # rotor distance
D = 0.23345235059857505

thrusts_to_T_M = np.array((
    (1,                1,               1,                1),
    (0,                -D,              0,                D),
    (D,                0,               -D,               0),
    (-torque_constant, torque_constant, -torque_constant, torque_constant)), dtype=float)

T_M_to_thrusts = np.linalg.inv(thrusts_to_T_M)

rotors_positions_wrt_body = [
    np.asarray((-0.14, -0.18, 0.05)),
    np.asarray((-0.14, 0.18, 0.05)),
    np.asarray((0.14, 0.18, 0.08)),
    np.asarray((0.14, -0.18, 0.08)),
]

sim_to_control_R = rotation_matrix("y", np.pi) @ rotation_matrix("z", -np.pi/4)

control_inertia = np.diag(sim_to_control_R @ inertia)

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


def get_control_drone_state(d, robot_body_id):
    global sim_to_control_R
    
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
        sim_to_control_R @ np.array([x0, y0, z0], dtype=float),
        sim_to_control_R @ np.array([vx0, vy0, vz0], dtype=float),
        np.asarray(sim_to_control_R @ R0, dtype=float).reshape(-1),
        sim_to_control_R @ np.array([omegax0, omegay0, omegaz0], dtype=float),
    ))

# Target

R_d = sim_to_control_R @ rotation_matrix("z", np.pi/2) # np.eye(3, dtype=float)
p_d = sim_to_control_R @ np.array((0,0,1), dtype=float)

g = np.array((0,0,-9.81), dtype=float)

Rc_previous = None
omega_c_previous = None

# controller gains

k_x = 20
k_v = 10

k_R = 5
k_omega = 0.5

DT = m.opt.timestep


def draw_custom_geometries(
    scene,
    robot_state,
    thrust_commands,
    show_collision_spheres,
):
    position = robot_state[:3]
    drone_orientation = robot_state[6:15].reshape(3, 3)
    thrusts_norm = np.linalg.norm(thrust_commands)
    speed = np.linalg.norm(robot_state[3:6])
    arrow_length = 0.15

    # Acceleration command arrow.
    if thrusts_norm > 1e-9:

        for i in range(4):
            motor_position = np.asarray(get_3d_site_position(
                d, bindings["robot1"]["sites"]["".join(["thrust", str(i+1)])]), dtype=float)

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
        robot_state[3:6] * arrow_length,
        [0.0, 1.0, 0.0, 0.8],
    )

    # control body frame
    draw_vector(
        scene,
        position,
        (drone_orientation @ sim_to_control_R)[:, 0],
        [1.0, 0.0, 0.0, 0.8],
    )
    draw_vector(
        scene,
        position,
        (drone_orientation @ sim_to_control_R)[:, 1],
        [0.0, 1.0, 0.0, 0.8],
    )
    draw_vector(
        scene,
        position,
        (drone_orientation @ sim_to_control_R)[:, 2], 
        [0.0, 0.0, 1.0, 0.8],
    )
    
    # Control inertial frame
    draw_vector(
        scene,
        np.zeros((3,1)),
        sim_to_control_R[:, 0],
        [1.0, 0.0, 0.0, 0.8],
    )
    draw_vector(
        scene,
        np.zeros((3,1)),
        sim_to_control_R[:, 1],
        [0.0, 1.0, 0.0, 0.8],
    )
    draw_vector(
        scene,
        np.zeros((3,1)),
        sim_to_control_R[:, 2], 
        [0.0, 0.0, 1.0, 0.8],
    )
    
    # desired position and attitude

    # Collision spheres.
    if show_collision_spheres:
        obstacles = get_collision_spheres(
            ["skydio_x2"], robot_body_name="skydio_x2")

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

def vec_to_skew(x):
    return np.array([
        [0,    -x[2],    x[1]],
        [x[2],     0,   -x[0]],
        [-x[1], x[0],       0]])
    
def skew_to_vec(x):
    return np.array([x[2,1], x[0,2], x[1, 0]])


try:
    with mujoco.viewer.launch_passive(
        m,
        d,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:

        pb = Playback()
        step = 0

        body_id = bindings["robot1"]["bodies"]["skydio_x2"]

        mujoco.mj_forward(m, d)

        initial_state = get_drone_state(d, body_id)

        previous_position_sim = None

        while viewer.is_running():
            step_start = time.time()

            # Read state before applying the next command.
            robot_state = get_drone_state(d, body_id)
            control_robot_state = get_control_drone_state(d, body_id)
            # robot_state[6:15] = np.eye(3).reshape(-1)
            control_robot_R = control_robot_state[6:15].reshape(3, 3)
            control_robot_omega = control_robot_state[15:18]

            # if step % 5 == 0:
                #print("sim position: ")
                #print(robot_state[0:3])
                # print("R:")
                # print(sim_to_control_R)
                #print("control position: ")
                #print(control_position)
            #     control_world_plot.update(robot_state)

            if pb.step > 0:
                pb.step -= 1
            elif pb.paused:
                viewer.sync()
                time.sleep(0.05)
                continue
            
            # Geometric position controller
            pos_error = control_robot_state[0:3] - p_d
            vel_error = control_robot_state[3:6]

            e_3 = sim_to_control_R[:, 2]
            
            b_3_c = - (- k_x * pos_error - k_v * vel_error - mass * g @ e_3) / np.abs(k_x * pos_error - k_v * vel_error - mass * g @ e_3)
            b_1_d = np.array((1,0,0))
            b_1_c = - (np.cross(b_3_c, np.cross(b_3_c, b_1_d)))/(np.linalg.norm(np.cross(b_3_c, b_1_d)))
            R_c = np.column_stack((b_1_c, np.cross(b_3_c, b_1_c), b_3_c))

            dR_c = np.zeros((3, 3))
            omega_c = np.zeros(3)
            domega_c = np.zeros(3)
            
            if Rc_previous is not None:
                dR_c = (R_c - Rc_previous) / DT

                # Finite differences introduce a symmetric component.
                # Project onto the skew-symmetric matrices before applying vee().
                omega_c_hat = R_c.T @ dR_c
                omega_c_hat = 0.5 * (omega_c_hat - omega_c_hat.T)
                omega_c = skew_to_vec(omega_c_hat)

                if omega_c_previous is None:
                    domega_c = np.zeros(3)
                else:
                    domega_c = (
                        omega_c - omega_c_previous
                    ) / DT

            Rc_previous = R_c.copy()
            omega_c_previous = (
                omega_c.copy() if step > 0 else None
            )
            
            rotation_error = 0.5 * skew_to_vec(R_c.T @ control_robot_R - control_robot_R.T @ R_c)
            omega_error = control_robot_omega - control_robot_R.T @ R_c @ omega_c 

            T = (k_x * pos_error + k_v * vel_error + mass * g @ e_3) @ control_robot_R @ e_3
             
            M_controller = -k_R * rotation_error - k_omega * omega_error + np.cross(control_robot_omega, control_inertia @ control_robot_omega) - control_inertia @ (vec_to_skew(control_robot_omega) @ control_robot_R.T @ R_c @ omega_c - control_robot_R.T @ R_c @ domega_c)

            M = M_controller # np.array((0.0, 0.0, 0.0))
            # T = -0.00

            paper_moments = M
            command = np.concatenate(
                ([T], np.multiply(paper_moments, np.array((1, 1, -1)))))
                # ([T], paper_moments))

            thrust_commands = T_M_to_thrusts @ command

            d.ctrl[actuator_thrust1] = thrust_commands[0]
            d.ctrl[actuator_thrust2] = thrust_commands[1]
            d.ctrl[actuator_thrust3] = thrust_commands[2]
            d.ctrl[actuator_thrust4] = thrust_commands[3]
           
            if step % 20 == 0:
                print(f"Control gains: k_x={k_x}, k_v={k_v}, k_R={k_R}, k_omega={k_omega}")
                print(f"pos_error: {pos_error}")
                print(f"vel_error: {vel_error}")
                print(f"R_error: {rotation_error}")
                print(f"omega_error: {omega_error}")
                print(f"thrust commands: {thrust_commands}, T={T}, M={M_controller}")

            with viewer.lock():
                viewer.user_scn.ngeom = 0

                draw_custom_geometries(
                    scene=viewer.user_scn,
                    robot_state=robot_state,
                    thrust_commands=thrust_commands,
                    show_collision_spheres=False,
                )

            mujoco.mj_step(m, d)

            step += 1

            viewer.sync()

            remaining_time = DT - (time.time() - step_start)

            if remaining_time > 0.0:
                time.sleep(remaining_time)

except Exception:
    import traceback
    traceback.print_exc()
