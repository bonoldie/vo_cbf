import xml.etree.ElementTree as ET
import mujoco


class SceneBuilder:
    def __init__(self, base_xml_path):
        self.tree = ET.parse(base_xml_path)
        self.root = self.tree.getroot()

        # We need the global worldbody element, if it doesn't exists create it
        self.worldbody = self.root.find("worldbody")
        if self.worldbody is None:
            raise ValueError("Missing <worldbody>")

        self.counter = 0

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

    def add_box(self, pos, size=(0.05, 0.05, 0.05), rgba=(1, 0, 0, 1), mass=None):
        body = ET.SubElement(
            self.worldbody,
            "body",
            {
                "name": self._next_name("box"),
                "pos": self._vec(pos),
            },
        )

        geom = {
            "type": "box",
            "size": self._vec(size),
            "rgba": self._rgba(rgba),
        }

        if mass is not None:
            geom["mass"] = str(mass)

        ET.SubElement(body, "geom", geom)

        return body

    def add_sphere(self, pos, radius=0.05, rgba=(0, 0, 1, 1)):
        body = ET.SubElement(
            self.worldbody,
            "body",
            {
                "name": self._next_name("sphere"),
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

        return body

    def add_cylinder(self, pos, radius=0.05, height=0.1, rgba=(0, 1, 0, 1)):
        body = ET.SubElement(
            self.worldbody,
            "body",
            {
                "name": self._next_name("cylinder"),
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

        return body

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

        # optional debug
        print(xml)

        return mujoco.MjModel.from_xml_string(xml)