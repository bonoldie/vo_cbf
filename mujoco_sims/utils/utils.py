import numpy  as np
import mujoco

# Utils
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def get_2d_pose(d, body_id):
    x, y, _= d.xpos[body_id]

    qw, qx, qy, qz = d.xquat[body_id]

    yaw = np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )

    return x, y, yaw

def get_3d_position(d, body_id):
    x, y, z= d.xpos[body_id]

    return x, y, z

def get_3d_velocity(d, body_id):
    _, _, _, vx, vy, vz = d.cvel[body_id]

    return vx, vy, vz



def get_v_w(m, d, body_id):

    body_vel = np.zeros(6)
             
    mujoco.mj_objectVelocity(
        m,
        d,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        body_vel,
        0  # world frame
    )

    return np.linalg.norm(body_vel[3:]), np.linalg.norm(body_vel[:3])

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

def draw_sphere(scene, pos, color=(1, 0, 0, 1), size=0.01):
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


