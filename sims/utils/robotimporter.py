import xml.etree.ElementTree as ET
import copy
import mujoco


class RobotImporter:
    def __init__(self, prefix=""):
        self.prefix = prefix

        # mapping tables
        self.map_body = {}
        self.map_joint = {}
        self.map_actuator = {}
        self.map_geom = {}
        self.map_site = {}
        self.map_light = {}

    # --------------------------------------------------
    # renaming helper
    # --------------------------------------------------

    def _p(self, name):
        if name is None:
            return None
        return f"{self.prefix}{name}"
    
    def _vec(self, v):
        return " ".join(map(str, v))
    

    # --------------------------------------------------
    # recursive renamer
    # --------------------------------------------------

    def _rename_element(self, el):
        tag = el.tag

        # rename known named objects
        if "name" in el.attrib:
            old = el.attrib["name"]
            new = self._p(old)
            el.attrib["name"] = new

            if tag == "body":
                self.map_body[old] = new
            elif tag == "joint":
                self.map_joint[old] = new
            elif tag == "geom":
                self.map_geom[old] = new
            elif tag == "site":
                self.map_site[old] = new
            elif tag == "light":
                self.map_light[old] = new
            elif tag == "velocity":
                self.map_actuator[old] = new

        # rename joint references inside actuators
        if tag == "velocity" and "joint" in el.attrib:
            el.attrib["joint"] = self._p(el.attrib["joint"])

        # recurse
        for child in el:
            self._rename_element(child)

    # --------------------------------------------------
    # main loader
    # --------------------------------------------------

    def load(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        robot_body = root.find("body")
        robot_actuator = root.find("actuator")

        if robot_body is None:
            raise ValueError("Robot XML must contain <body>")

        # deep copy so we don't mutate original
        body = copy.deepcopy(robot_body)
        actuator = copy.deepcopy(robot_actuator) if robot_actuator is not None else None

        # rename everything
        self._rename_element(body)
        if actuator is not None:
            self._rename_element(actuator)

        return body, actuator



    # --------------------------------------------------
    # public API
    # --------------------------------------------------

    def import_robot(self, model_root, path, pos=None, euler=None ):
        """
        model_root = mujoco XML root (ET root)
        """
        worldbody = model_root.find("worldbody")
        actuator_root = model_root.find("actuator")

        if worldbody is None:
            raise ValueError("Missing worldbody")

        # Here body must be the robot structure
        body, actuator = self.load(path)

        # Robot params override
        if pos is not None:
            body.attrib["pos"] = self._vec(pos)

        if euler is not None:
            body.attrib["euler"] = self._vec(euler)

        # inject into scene
        worldbody.append(body)

        if actuator is not None:
            if actuator_root is None:
                actuator_root = ET.SubElement(model_root, "actuator")

            for a in actuator:
                actuator_root.append(a)

        return {
            "body": self.map_body,
            "joints": self.map_joint,
            "actuators": self.map_actuator,
            "geom": self.map_geom,
            "sites": self.map_site,
            "lights": self.map_light,
        }



def resolve_robot_ids(model, robot_maps):
    """
    Converts name maps -> mujoco IDs once model is built.
    """

    def safe_id(obj_type, name):
        if name is None:
            return None
        return mujoco.mj_name2id(model, obj_type, name)

    return {
        "bodies": {k: safe_id(mujoco.mjtObj.mjOBJ_BODY, v)
                 for k, v in robot_maps["body"].items()},

        "joints": {k: safe_id(mujoco.mjtObj.mjOBJ_JOINT, v)
                   for k, v in robot_maps["joints"].items()},

        "actuators": {k: safe_id(mujoco.mjtObj.mjOBJ_ACTUATOR, v)
                      for k, v in robot_maps["actuators"].items()},

        "geom": {k: safe_id(mujoco.mjtObj.mjOBJ_GEOM, v)
                 for k, v in robot_maps["geom"].items()},

        "sites": {k: safe_id(mujoco.mjtObj.mjOBJ_SITE, v)
                 for k, v in robot_maps["sites"].items()},
    }