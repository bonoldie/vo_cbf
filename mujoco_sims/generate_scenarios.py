import os
import csv
import numpy as np
import mujoco
import mujoco.viewer
from  utils.scenebuilder import SceneBuilder
from  utils.robotimporter import RobotImporter, resolve_robot_ids

def buildModel(
        # Robot example: {"name": "robot_1", "pos": (0, -1, 0.05), "euler": (0, 0, np.pi), "collision_radius": 0.3, "robot_path": "path/to/robot.xml"}
        robots=[], 
        # Obstacle examples:  
        #    - {"type": ObstacleType.SPHERE, "pos": (1, 1, 1), "radius": 0.25}
        #    - {"type": ObstacleType.SPHERE, "pos": (2, 2, 2), "radius": 0.25}
        obstacles=[],
        prebuild_hook = lambda x: None, 
        base_path = None, 
        assets_path = None, 
        defaults_path = None, 
        worldbody_path = None,  
    ):
    """
    Build the model of the scene with a given number of robots and obstacles 
    """

    if base_path is None:
        raise Exception('No base xml path specified, check your base_path and retry')

    builder = SceneBuilder(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        base_path
    ))

    if assets_path is not None:
        builder.include_xml_asset(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            assets_path
        ))

    if defaults_path is not None:
        builder.include_xml_default(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            defaults_path
        ))

    if worldbody_path is not None:
        builder.include_xml_worldbody(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            worldbody_path
        ))

    builder.add_obstacles(obstacles)

    mappings = {}

    for robot in robots:
        if robot["name"] is None:
            raise Exception("Car must have a name at least")

        if  robot["collision_radius"] is None:
            raise Exception("collision_radius property not found in robot definition")


        robot_importer = RobotImporter(prefix=robot["name"]+"_")

        robot_mapping = robot_importer.import_robot(
            builder.root,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), robot["robot_path"]),
            pos=robot["pos"] if "pos" in robot.keys() else None,
            euler=robot["euler"] if "euler" in robot.keys() else None,
        )
        
        mappings[robot["name"]] = robot_mapping

    prebuild_hook(builder)

    m = builder.build_model()
    d = mujoco.MjData(m)

    ids_mappings = [[car_name, resolve_robot_ids(m, robot_mapping)] for car_name, robot_mapping in mappings.items()]
    
    bindings = dict(ids_mappings)

    def get_collision_spheres(blacklist=[]):
        # Here get the current cars and obstacles position, velocity and acceleration, return a dict with { obstacle_name: [p: [px, py,pz], v:[vx, vy, vz]]}
        obstacles = {}

        for robot in filter(lambda x : x["name"] not in blacklist, robots):
            obstacles[robot["name"]] = {}
            obstacles[robot["name"]]["p"] = d.xpos[
                bindings[robot["name"]]["bodies"]["robot"]
            ]

            obstacle_vel = np.zeros(6)
             
            mujoco.mj_objectVelocity(
                m,
                d,
                mujoco.mjtObj.mjOBJ_BODY,
                bindings[robot["name"]]["bodies"]["robot"],
                obstacle_vel,
                0  # world frame
            )

            obstacles[robot["name"]]["v"] = obstacle_vel[3:]

            # save collision radius for later computations
            obstacles[robot["name"]]["collision_radius"] = robot["collision_radius"]
        
        for obstacle_name in filter(lambda x : x not in blacklist, builder.obstacles.keys()) :
            obstacle_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obstacle_name)

            obstacles[obstacle_name] = {}
            obstacles[obstacle_name]["p"] = d.xpos[
                obstacle_body_id
            ]

            obstacle_vel = np.zeros(6)
             
            mujoco.mj_objectVelocity(
                m,
                d,
                mujoco.mjtObj.mjOBJ_BODY,
                obstacle_body_id,
                obstacle_vel,
                0  # world frame
            )

            obstacles[obstacle_name]["v"] = obstacle_vel[3:]
            obstacles[obstacle_name]["collision_radius"] = builder.obstacles[obstacle_name]["collision_radius"]

        return obstacles

    return m, d, bindings, get_collision_spheres


def format_obstacles(obstacles):
    lines = []
    lines.append("Obstacles state:")

    for name, data in obstacles.items():
        p = data["p"]
        v = data["v"]
        collision_radius = data["collision_radius"]

        # handle numpy arrays safely
        p = list(p)
        v = list(v)

        lines.append(
            f"  {name} (collision radius: {collision_radius:1.3f}): p = [{p[0]: .3f}, {p[1]: .3f}, {p[2]: .3f}] v = [{v[0]: .3f}, {v[1]: .3f}, {v[2]: .3f}]"
        )

    return "\n".join(lines)