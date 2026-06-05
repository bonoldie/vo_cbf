import os
import csv
import numpy as np
import mujoco
import mujoco.viewer
from  utils.scenebuilder import SceneBuilder
from  utils.robotimporter import RobotImporter, resolve_robot_ids

def buildModel(cars=[{"name": "car_1", "pos": (0, -1, 0.05), "euler": (0, 0, np.pi), "collision_radius": 0.3}], obstacles=[],prebuild_hook = lambda x: None ):
    """
    Build the model of the scene with a given number of robots and obstacles 
    """
    builder = SceneBuilder(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scenarios/multicar/base.xml"
    ))

    builder.include_xml_asset(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scenarios/multicar/assets.xml"
    ))

    builder.include_xml_default(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scenarios/multicar/defaults.xml"
    ))

    builder.include_xml_worldbody(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scenarios/multicar/world.xml"
    ))

    builder.add_obstacles(obstacles)

    mappings = {}

    for car in cars:
        if car["name"] is None:
            raise Exception("Car must have a name at least")

        if  car["collision_radius"] is None:
            raise Exception("collision_radius property not found in robot definition")


        robot_importer = RobotImporter(prefix=car["name"]+"_")

        robot_mapping = robot_importer.import_robot(
            builder.root,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios/multicar/car.xml"),
            pos=car["pos"] if "pos" in car.keys() else None,
            euler=car["euler"] if "euler" in car.keys() else None,
        )

        mappings[car["name"]] = robot_mapping

    prebuild_hook(builder)

    m = builder.build_model()
    d = mujoco.MjData(m)

    ids_mappings = [[car_name, resolve_robot_ids(m, robot_mapping)] for car_name, robot_mapping in mappings.items()]
    
    bindings = dict(ids_mappings)

    def get_collision_spheres():
        # Here get the current cars and obstacles position, velocity and acceleration, return a dict with { obstacle_name: [p: [px, py,pz], v:[vx, vy, vz]]}
        obstacles = {}

        for car in cars:
            obstacles[car["name"]] = {}
            obstacles[car["name"]]["p"] = d.xpos[
                bindings[car["name"]]["bodies"]["car"]
            ]

            obstacle_vel = np.zeros(6)
             
            mujoco.mj_objectVelocity(
                m,
                d,
                mujoco.mjtObj.mjOBJ_BODY,
                bindings[car["name"]]["bodies"]["car"],
                obstacle_vel,
                0  # world frame
            )

            obstacles[car["name"]]["v"] = obstacle_vel[3:]

            # save collision radius for later computations
            obstacles[car["name"]]["collision_radius"] = car["collision_radius"]
        
        for obstacle_name in builder.obstacles.keys():
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
            f"  {name} (collision radius: {collision_radius}):\n"
            f"    p = [{p[0]: .3f}, {p[1]: .3f}, {p[2]: .3f}]\n"
            f"    v = [{v[0]: .3f}, {v[1]: .3f}, {v[2]: .3f}]"
        )

    return "\n".join(lines)