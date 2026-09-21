import time
import numpy as np
import mujoco
import mujoco.viewer

from generate_scenarios import buildModel, format_obstacles
from utils.playback import Playback
from utils.scenebuilder import ObstacleType
from utils.utils import *
from controllers.qp_3d_precomp_drone import QP3DPrecompDrone


target = np.array([4.0, 2.0, 3.0])
target_side = 1
controller = None

sh_n = 6
sh_tau = 1.6

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
        grid_size=(2, 2),
    density=2,
    cell_size=1,
    z_range=(0.5, 1.5),
    seed=44,
)


def generate_new_target(margin=0.5):
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


m, d, bindings, get_collision_spheres = buildModel(
    [
        {
            "name": "robot",
            "collision_radius": 0.3,
            "pos": (0, 0, 0),
            "robot_path": "scenarios/drone/skydio_x2.xml"
        }
    ],
    obstacles,
    base_path="scenarios/drone/base.xml",
    worldbody_path="scenarios/drone/world.xml",
    assets_path="scenarios/drone/assets.xml",
    defaults_path="scenarios/drone/defaults.xml"
)


# W maps simulation-world coordinates to control-world coordinates.
# B maps simulation-body coordinates to control-body coordinates.
# They are separate changes of basis, even though we choose equal matrices.
W = rotation_matrix("y", np.pi) @ rotation_matrix("z", -np.pi / 4)
B = W.copy()

def rigid_properties(m, d, body_id):
    """Mass, COM offset and inertia of a rigid subtree in its root body frame."""
    members = [body_id]
    
    for i in range(body_id + 1, m.nbody):
        if int(m.body_parentid[i]) in members:
            if m.body_dofnum[i] != 0:
                raise ValueError("Controller requires a rigid drone subtree.")
            members.append(i)
    
    # Summing the masses of of the bodies under the root drone frame        
    total_mass = float(np.sum(m.body_mass[members]))
    
    if total_mass <= 0:
        raise ValueError("Drone mass must be positive.")

    # body-referenced COM
    com_body = m.body_ipos[body_id]
    
    # body-reference Inertia
    J_body = np.diag(m.body_inertia[body_id])
    
    return total_mass, com_body, J_body, members

def read_state(m, d, body_id, W, B, com_body):
    """Position/velocity in control world; angular velocity in control body."""
    Rsim = d.xmat[body_id].reshape(3, 3)
    com_world = d.xpos[body_id] + Rsim @ com_body
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, com_world, body_id)
    v_world = jp @ d.qvel
    omega_world = jr @ d.qvel
    return (
        W @ com_world,
        W @ v_world,
        W @ Rsim @ B.T,
        B @ Rsim.T @ omega_world,
    )


def build_mixer(m, d, body_id, actuator_ids, B, com_body, members):
    """Map direct motor controls to [positive thrust, control-body moments]."""
    Rsim = d.xmat[body_id].reshape(3, 3)
    
    # Sim inertial-referenced COM
    com_world = d.xpos[body_id] + Rsim @ com_body
    
    allocation = np.empty((4, len(actuator_ids)))
    
    for col, aid in enumerate(actuator_ids):
        # Sanity check
        if m.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_SITE:
            raise ValueError("Expected site-transmission rotor motors.")
        if (m.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE or m.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED or m.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_NONE):
            raise ValueError("Expected direct motors with fixed gain and no bias.")
        
        # actuator transmission ID: joint, tendon, site
        sid = int(m.actuator_trnid[aid, 0])
        
        # Sanity check
        if m.actuator_trnid[aid, 1] != -1:
            raise ValueError("Rotor motors must not use refsite.")
        if int(m.site_bodyid[sid]) not in members:
            raise ValueError("Rotor site is not attached to this rigid drone.")
        
        # Actuator frame
        # Cartesian site orientation
        Rsite = d.site_xmat[sid].reshape(3, 3)
        gain = float(m.actuator_gainprm[aid, 0])
        
        # Sim inertial-referenced force and torques generated by the actuator
        force_world = Rsite @ m.actuator_gear[aid, :3] * gain
        
        # Torque given by the spinning actuator 
        torque_world = Rsite @ m.actuator_gear[aid, 3:] * gain
        torque_world += np.cross(d.site_xpos[sid] - com_world, force_world)
         
        force_body = B @ Rsim.T @ force_world
        torque_body = B @ Rsim.T @ torque_world
        
        # Sanity check    
        if not np.allclose(force_body[:2], 0.0, atol=1e-8) or force_body[2] >= 0:
            raise ValueError("Positive motor force must point along control-body -z.")
        
        allocation[0, col] = -force_body[2]
        allocation[1:, col] = torque_body
    
    if np.linalg.matrix_rank(allocation) != 4:
        raise ValueError("Rotor allocation is rank deficient: check XML rotor positions and yaw gear signs.")

    return allocation


def load_control_limits(m, actuator_ids):
    """Respect control and scalar actuator-force limits for these direct motors."""
    lower, upper = np.zeros(4), np.full(4, np.inf)
    
    for i, aid in enumerate(actuator_ids):
        gain = float(m.actuator_gainprm[aid, 0])
        if gain <= 0:
            raise ValueError("Expected positive motor gains.")
        if m.actuator_ctrllimited[aid]:
            lower[i] = max(lower[i], m.actuator_ctrlrange[aid, 0])
            upper[i] = min(upper[i], m.actuator_ctrlrange[aid, 1])
        if m.actuator_forcelimited[aid]:
            lower[i] = max(lower[i], m.actuator_forcerange[aid, 0] / gain)
            upper[i] = min(upper[i], m.actuator_forcerange[aid, 1] / gain)
    
    if np.any(lower >= upper):
        raise ValueError("Inconsistent rotor limits.")
    
    return lower, upper


def position_controller(p, v, R, omega, p_d, gravity, mass, J,
                        k_x=10.0, k_v=5.0, k_R=5.0, k_omega=0.5):
    A = -k_x * (p - p_d) - k_v * v - mass * gravity
    norm_A = np.linalg.norm(A)
    if norm_A < 1e-9:
        raise ValueError("Zero desired force: desired attitude is undefined.")
    b3 = -A / norm_A
    b2 = np.cross(b3, np.array([1.0, 0.0, 0.0]))
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-9:
        raise ValueError("Desired heading is parallel to thrust direction.")
    b2 /= norm_b2
    b1 = np.cross(b2, b3)
    Rc = np.column_stack((b1, b2, b3))
    e_R = 0.5 * vee(Rc.T @ R - R.T @ Rc)
    # Feedforward omega_c and domega_c are disabled for this position-hold baseline.
    M = -k_R * e_R - k_omega * omega + np.cross(omega, J @ omega)
    T = -float(A @ R[:, 2])
    angle = np.arccos(np.clip((np.trace(Rc.T @ R) - 1) / 2, -1, 1))
    return T, M, Rc, e_R, float(angle)

# Main body of the drone, we will load the rigid body props from this element
body_id = bindings["robot"]["bodies"]["skydio_x2"]

# Actuators ids
actuator_ids = np.array([
    bindings["robot"]["actuators"][f"thrust{i}"] for i in range(1, 5)
], dtype=int)

mujoco.mj_forward(m, d)

# Rigid body properties
mass, com_body, J_body, members = rigid_properties(m, d, body_id)

control_inertia = B @ J_body @ B.T
assert np.min(np.linalg.eigvalsh(control_inertia)) > 0

allocation = build_mixer(m, d, body_id, actuator_ids, B, com_body, members)
T_M_to_thrusts = np.linalg.inv(allocation)
lower, upper = load_control_limits(m, actuator_ids)

# Target position 
p_d = W @ target
gravity_world = W @ np.asarray(m.opt.gravity)
DT = float(m.opt.timestep)

# Print info
print("Drone mass:", mass)
print("Control inertia:\n", control_inertia)
print("Actual allocation [T; M] = allocation @ motor_controls:\n", allocation)
print("Motor limits:", lower, upper)


# Controller setup
# needed for initial state
p, v, R, omega = read_state(m, d, body_id, W, B, com_body)

controller = QP3DPrecompDrone(
    dt=DT,
    target=target,
    initial_state=np.vstack((p, v)),
    collision_radius=0.3,
    sh_n=sh_n,
    sh_tau=sh_tau,
    obstacles=get_collision_spheres(["robot"]),
    mass = mass
)


try:
    pb = Playback()
    step = 0
    with mujoco.viewer.launch_passive(
        m, d, show_left_ui=True, show_right_ui=True
    ) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            if pb.step > 0:
                pb.step -= 1
            elif pb.paused:
                viewer.sync()
                time.sleep(0.05)
                continue


            # Refresh derived poses/velocities for the current qpos/qvel.
            mujoco.mj_forward(m, d)
            p, v, R, omega = read_state(m, d, body_id, W, B, com_body)
            T, M, Rc, e_R, angle = position_controller(p, v, R, omega, p_d, gravity_world, mass, control_inertia)
            command = np.concatenate(([T], M))
            controls_requested = T_M_to_thrusts @ command
            controls_applied = np.clip(controls_requested, lower, upper)
            if not np.all(np.isfinite(controls_applied)):
                raise FloatingPointError("Non-finite motor command.")
            d.ctrl[actuator_ids] = controls_applied

            if step % max(1, round(0.1 / DT)) == 0:
                assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
                assert np.allclose(Rc.T @ Rc, np.eye(3), atol=1e-6)
                print(
                    f"t={d.time:.2f}, pos_error={p-p_d}, "
                    f"attitude_angle_deg={np.degrees(angle):.3f}, omega={omega}"
                )
                print(f"T={T:.3f}, M={M}, controls={controls_applied}")
                if not np.allclose(controls_requested, controls_applied):
                    print("Actuator saturation; achieved [T; M]:", allocation @ controls_applied)

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                Rsim = d.xmat[body_id].reshape(3, 3)
                com_world = W.T @ p
                colors = ([1, 0, 0, 0.8], [0, 1, 0, 0.8], [0, 0, 1, 0.8])
                for i, aid in enumerate(actuator_ids):
                    sid = int(m.actuator_trnid[aid, 0])
                    Rsite = d.site_xmat[sid].reshape(3, 3)
                    force = (Rsite @ m.actuator_gear[aid, :3]
                             * m.actuator_gainprm[aid, 0] * controls_applied[i])
                    draw_vector(viewer.user_scn, d.site_xpos[sid], 0.15 * force, [1, 1, 0, 0.8])
                draw_vector(viewer.user_scn, com_world, 0.15 * (W.T @ v), [0, 1, 1, 0.8])
                # Control-body axes drawn in simulation world.
                axes = Rsim @ B.T
                for i, color in enumerate(colors):
                    draw_vector(viewer.user_scn, com_world, 0.3 * axes[:, i], color)
                    draw_vector(viewer.user_scn, np.zeros(3), 0.3 * W.T[:, i], color)

            mujoco.mj_step(m, d)
            step += 1
            viewer.sync()
            remaining_time = DT - (time.perf_counter() - step_start)
            if remaining_time > 0:
                time.sleep(remaining_time)
except KeyboardInterrupt:
    pass
