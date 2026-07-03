from setuptools import find_packages, setup

import os
from glob import glob


package_name = "double_integrator_2d"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "rviz"),
            glob("rviz/*.rviz"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="enrico",
    maintainer_email="enrico.bonoldi@univr.it",
    description="2D double-integrator simulator with RViz markers and obstacle TFs",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim = double_integrator_2d.sim_node:main",
            "obstacles = double_integrator_2d.obstacles_node:main",
            "controller = double_integrator_2d.controller_node:main",
            "cbf_controller = double_integrator_2d.cbf_controller_node:main"
        ],
    },
)