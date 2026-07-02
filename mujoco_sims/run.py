import os
import csv
import numpy as np
import mujoco
import mujoco.viewer
from generate_scenarios import buildModel
from scipy.optimize import minimize



def add_obstacles(builder):
    builder.add_box(
        pos=(1, 0, 0.1),
        size=(0.1, 0.1, 0.1)
    )

    builder.add_cylinder(
        pos=(0, 1, 0.05),
        radius=0.05
    )

m, d, bindings = buildModel([{
    "name": "car_1"
}, {
    "name": "car_2",
    "pos": (0,-1,0.05)
}], add_obstacles)


left_actuator_id = bindings["car_1"]["actuators"]["left_wheel"]
right_actuator_id = bindings["car_1"]["actuators"]["right_wheel"]

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------

target = np.array([1.0, 1.0])


# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------

wheel_radius = 0.015
wheel_base = 0.02

MAX_V = 0.04
MAX_W = 0.3
STOP_DIST = 0.01


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def get_pose():
    x, y, _= d.xpos[bindings['car_1']["bodies"]["car"]]

    qw, qx, qy, qz = d.xquat[bindings['car_1']["bodies"]["car"]]

    yaw = np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )

    return x, y, yaw


def Rz(v):
    v = v/np.linalg.norm(v)
    z = np.array([0.,0.,1.])
    if np.allclose(v,z): return np.eye(3)
    if np.allclose(v,-z): return np.diag([1,-1,-1])

    a = np.cross(z,v); a /= np.linalg.norm(a)
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    c = np.dot(z,v); s = np.linalg.norm(np.cross(z,v))

    return np.eye(3) + s*K + (1-c)*(K@K)

def draw_vector(scene, start, vec, color):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=[0.005, 0.005, np.linalg.norm(vec)],
        pos=start + [0, 0, 0.05],
        mat=Rz(vec).flatten(),
        rgba=color
    )
    scene.ngeom += 1

def draw_point(scene, pos, color=(1, 0, 0, 1), size=0.01):

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[size, 0, 0],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=color
    )

    scene.ngeom += 1

# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------
DT = m.opt.timestep

def rollout(pos, yaw, v, w, dt):
    x = pos[0] + v * np.cos(yaw) * dt
    y = pos[1] + v * np.sin(yaw) * dt
    yaw2 = yaw + w * dt
    return np.array([x, y]), yaw2


def controller(pos, yaw, target):

    desired_yaw = np.arctan2(
        target[1] - pos[1],
        target[0] - pos[0]
    )

    def cost(u):

        v, w = u

        # predict future pose
        p2, yaw2 = rollout(pos, yaw, v, w, DT * 15)

        # target errors
        pos_err = np.linalg.norm(target - p2)

        yaw_err = wrap(desired_yaw - yaw2)

        # objective
        J = (
            5.0 * pos_err**2 +
            2.5 * yaw_err**2 +
            0.1 * v**2 +
            0.05 * w**2
        )

        return J

    res = minimize(
        cost,
        x0=[0.0, 0.0],
        bounds=[
            (-MAX_V, MAX_V),
            (-MAX_W, MAX_W)
        ],
        method="SLSQP"
    )

    v, w = res.x

    dist = np.linalg.norm(target - pos)
    yaw_err = wrap(desired_yaw - yaw)

    if dist < STOP_DIST:
        v = 0.0
        w = 0.0

    return v, w, dist, yaw_err

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

log = []


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

with mujoco.viewer.launch_passive(m, d) as viewer:

    step = 0

    while viewer.is_running():

        x, y, yaw = get_pose()
        pos = np.array([x, y])

        v, w, dist, yaw_err = controller(pos, yaw, target)

        # differential drive
        vl = (v - 0.5 * wheel_base * w) / wheel_radius
        vr = (v + 0.5 * wheel_base * w) / wheel_radius

        vl = np.clip(vl, -15, 15)
        vr = np.clip(vr, -15, 15)

        d.ctrl[left_actuator_id] = vl
        d.ctrl[right_actuator_id] = vr

        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0
            pos = np.array([x, y, 0.05])

            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])

            # scaled linear velocity vector
            v_vec = forward * v
            draw_vector(scene, pos, v_vec * 10, [1, 1, 0, 0.8 if np.linalg.norm(v_vec) > 0.0005 else 0.2])      # linear velocity (yellow)
            draw_vector(scene, pos, np.array([0, 0, w]), [1, 0, 0, 0.8 if abs(w) > 0.01 else 0.2])      # angular velocity (yellow)

            draw_point(scene, np.append(target, 0.05), (1, 0, 0, 1), 0.03)

        mujoco.mj_step(m, d)
        viewer.sync()

        # -----------------------------------------------------------------------------
        # LOGGING (structured + readable)
        # -----------------------------------------------------------------------------

        log.append([d.time, x, y, yaw, v, w, dist, yaw_err])

        # print every 20 steps (prevents spam)
        if step % 20 == 0:
            print(
                f"[t={d.time:5.2f}] "
                f"pos=({x: .2f},{y: .2f}) "
                f"dist={dist: .3f} "
                f"yaw_err={yaw_err: .2f} "
                f"v={v: .3f} w={w: .3f} "
                f"vl={vl: .2f} vr={vr: .2f}"
            )

        step += 1


# -----------------------------------------------------------------------------
# Save log
# -----------------------------------------------------------------------------

with open("tb3_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t", "x", "y", "yaw", "v", "w", "dist", "yaw_err"])
    writer.writerows(log[::50])