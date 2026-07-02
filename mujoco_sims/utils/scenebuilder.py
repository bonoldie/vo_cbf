import xml.etree.ElementTree as ET
import mujoco
from enum import Enum
from pathlib import Path

class ObstacleType(Enum):
    SPHERE=0
    BOX=1
    CYLINDER=2


class SceneBuilder:


    def __init__(self, base_xml_path):
        self.tree = ET.parse(base_xml_path)
        self.root = self.tree.getroot()

        # We need the global worldbody element, if it doesn't exists create it
        self.worldbody = self.root.find("worldbody")
        if self.worldbody is None:
            raise ValueError("Missing <worldbody>")

        # Init temp dir
        tmp_dir = Path(__file__).resolve().parent.parent.parent / ".tmp"
        tmp_dir.mkdir(exist_ok=True)
        self.built_scene_xml_path = tmp_dir / "built_scene.xml"

        self.counter = 0

        self.obstacles = {}
    # --------------------------------------------------
    # helpers
    # --------------------------------------------------

    def _next_name(self, prefix):
        name = f"{prefix}_{self.counter}"
        self.counter += 1
        return name

    def _vec(self, v):
        return " ".join(map(str, v))

    def _rgba(self, rgba):
        return " ".join(map(str, rgba))

    # --------------------------------------------------
    # obstacles
    # --------------------------------------------------

    def add_box(self, pos, size=(0.05, 0.05, 0.05), rgba=(1, 0, 0, 1), name_prefix="box"):
        box_name = self._next_name(name_prefix)

        body = ET.SubElement(
            self.worldbody,
            "body",
            {
                "name": box_name,
                "pos": self._vec(pos),
            },
        )

        geom = {
            "type": "box",
            "size": self._vec(size),
            "rgba": self._rgba(rgba),
        }

        ET.SubElement(body, "geom", geom)

        return body, box_name

    def add_sphere(self, pos, radius=0.05, rgba=(0, 0, 1, 1), name_prefix="sphere"):
        sphere_name = self._next_name(name_prefix)

        body = ET.SubElement(
            self.worldbody,
            "body",
            {
                "name": sphere_name,
                "pos": self._vec(pos),
            },
        )

        ET.SubElement(
            body,
            "geom",
            {
                "type": "sphere",
                "size": str(radius),
                "rgba": self._rgba(rgba),
            },
        )

        return body, sphere_name

    def add_cylinder(self, pos, radius=0.05, height=0.1, rgba=(0, 1, 0, 1), name_prefix="cylinder"):
        cylinder_name = self._next_name(name_prefix)

        body = ET.SubElement(
            self.worldbody,
            "body",
            {
                "name": cylinder_name,
                "pos": self._vec(pos),
            },
        )

        # MuJoCo expects: radius + half-height
        ET.SubElement(
            body,
            "geom",
            {
                "type": "cylinder",
                "size": f"{radius} {height / 2.0}",
                "rgba": self._rgba(rgba),
            },
        )

        return body, cylinder_name

    def add_obstacle(self, obstacle: dict):
        """
        Add obstacle to the scene, does obstacle definition validation also 
        """

        # Obstacle definition validation
        if "pos" not in obstacle.keys():
            raise Exception("Obstacle must have a position")            

        match obstacle["type"]:
            case ObstacleType.SPHERE:
                _, name = self.add_sphere(**{
                    k: obstacle[k] for k in ("pos", "radius", "rgba") if k in obstacle
                })

                self.obstacles[name] = obstacle
                self.obstacles[name]["collision_radius"] = self._resolve_obstacle_radius(obstacle)


            case ObstacleType.CYLINDER:
                _, name = self.add_cylinder(**{
                    k: obstacle[k] for k in ("pos", "radius", "height", "rgba") if k in obstacle
                })

                self.obstacles[name] = obstacle
                self.obstacles[name]["collision_radius"] = self._resolve_obstacle_radius(obstacle)

            case ObstacleType.BOX:
                _, name = self.add_box(**{
                    k: obstacle[k] for k in ("pos", "size", "rgba") if k in obstacle
                })

                self.obstacles[name] = obstacle
                self.obstacles[name]["collision_radius"] = self._resolve_obstacle_radius(obstacle)
            case _:
                raise Exception("Give obstacle type not implemented")

        return

    def add_obstacles(self, obstacles=[]):
        for obstacle in obstacles:
            self.add_obstacle(obstacle) 


    def _resolve_obstacle_radius(self, obstacle):
        """
        Returns a scalar 'effective radius' for any obstacle type.
        """
        if "obstacle_radius" in obstacle:
            return obstacle["obstacle_radius"]

        match obstacle["type"]:
            case ObstacleType.SPHERE:
                return obstacle.get("radius", 0.05)

            case ObstacleType.CYLINDER:
                r = obstacle.get("radius", 0.05)
                h = obstacle.get("height", 0.1)
                return max(r, h / 2.0)

            case ObstacleType.BOX:
                size = obstacle.get("size", (0.05, 0.05, 0.05))
                # effective radius = half diagonal
                return (sum((s ** 2 for s in size)) ** 0.5)

            case _:
                raise Exception("Unknown obstacle type")
        
    # --------------------------------------------------
    # include other xml fragments safely
    # --------------------------------------------------

    def include_xml_worldbody(self, xml_path):
        """
        Safely inject only <worldbody> children from another file.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        wb = root
        if wb is None:
            raise ValueError(f"No worldbody in {xml_path}")

        for child in wb:
            self.worldbody.append(child)

    def include_xml_asset(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        asset = root
        if asset is None:
            return

        target_asset = self.root.find("asset")
        if target_asset is None:
            target_asset = ET.SubElement(self.root, "asset")

        for child in asset:
            target_asset.append(child)

    def include_xml_default(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        default = root
        if default is None:
            return

        target_default = self.root.find("default")
        if target_default is None:
            target_default = ET.SubElement(self.root, "default")

        for child in default:
            target_default.append(child)


    def include_xml_actuator(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        actuator = root
        if actuator is None:
            return

        target_actuator = self.root.find("actuator")
        if target_actuator is None:
            target_actuator = ET.SubElement(self.root, "actuator")

        for child in actuator:
            target_actuator.append(child)

    # --------------------------------------------------
    # build
    # --------------------------------------------------

    def xml_string(self):
        return ET.tostring(self.root, encoding="unicode")

    def build_model(self):
        xml = self.xml_string()

        # Saving the final scene debug purposes
        with open(self.built_scene_xml_path, "w") as f:
            f.write(xml)


        return mujoco.MjModel.from_xml_string(xml)