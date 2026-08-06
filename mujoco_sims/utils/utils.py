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


def get_3d_site_position(d, site_id):
    x, y, z= d.site_xpos[site_id]

    return x, y, z

def get_3d_orientation(d, body_id):
    return d.xmat[body_id].reshape(3, 3)

def get_3d_velocity(d, body_id):
    _, _, _, vx, vy, vz = d.cvel[body_id]

    return vx, vy, vz

def get_3d_angular_velocity(d, body_id):
    omega_x, omega_y, omega_z, _, _, _ = d.cvel[body_id]

    return omega_x, omega_y, omega_z


def get_mass(m, body_id):
    mass = m.body_mass[body_id]

    return mass


def get_inertia(m, body_id):
    inertia = m.body_inertia[body_id]

    return np.asarray(inertia)




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
        pos=start,
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

def rotation_matrix(axis, angle):
    """
    Return a 3x3 rotation matrix.

    Parameters
    ----------
    axis : str or array-like
        Either "x", "y", "z", or a 3D rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    if isinstance(axis, str):
        axes = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }

        try:
            axis = axes[axis.lower()]
        except KeyError:
            raise ValueError("Axis must be 'x', 'y', 'z', or a 3D vector.")
    else:
        axis = np.asarray(axis, dtype=float)

    if axis.shape != (3,):
        raise ValueError("The rotation axis must be a 3D vector.")

    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("The rotation axis cannot be zero.")

    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ])