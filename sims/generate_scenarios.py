import os
import csv
import numpy as np
import mujoco
import mujoco.viewer
from  utils.scenebuilder import SceneBuilder
from  utils.robotimporter import RobotImporter, resolve_robot_ids

def buildModel(cars=[{"name": "car_1", "pos": (0, -1, 0.05), "euler": (0, 0, np.pi)}], obstacles=[],prebuild_hook = lambda x: None ):
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


    return m, d, dict(ids_mappings)
