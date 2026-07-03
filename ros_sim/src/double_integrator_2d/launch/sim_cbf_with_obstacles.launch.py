from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Start RViz2",
    )

    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("double_integrator_2d"),
            "rviz",
            "double_integrator.rviz",
        ]
    )

    sim_node = Node(
        package="double_integrator_2d",
        executable="sim",
        name="double_integrator_2d_sim",
        output="screen",
        parameters=[
            {
                "dt": 0.01,
                "frame_id": "odom",
                "child_frame_id": "base_link",
                "robot_radius": 0.15,
                "max_accel": 2.0,
                "max_velocity": 5.0,
                "cmd_timeout": 0.5,
                "publish_tf": True,
            }
        ],
    )

    obstacles_node = Node(
        package="double_integrator_2d",
        executable="obstacles",
        name="obstacles_node",
        output="screen",
        parameters=[
            {
                "frame_id": "odom",
                "dt": 0.05,
                "path_history_size": 500,
            }
        ],
    )

    controller_node = Node(
        package="double_integrator_2d",
        executable="cbf_controller",
        name="cbf_controller_node",
        output="screen",
        parameters=[
            {
                "frame_id": "odom",
                "robot_frame_id": "base_link",
                "target_frame_id": "target",
                "obstacle_frames": [
                    "obs_fixed_1",
                    "obs_fixed_2",
                    "obs_circle_1",
                    "obs_line_1",
                ],

                "dt": 0.1,

                "target_x": 4.0,
                "target_y": 0.0,

                "max_accel": 0.5,
                "target_tolerance": 0.005,

                "mpc_horizon": 10,
                "mpc_dt": 0.1,

                "robot_radius": 0.15,
                "obstacle_radius_default": 0.25,

                "cruising_speed": 0.6,

                "q_terminal": 40.0,
                "q_position": 2.0,
                "q_heading": 2.0,
                "q_cruising_speed": 4.0,

                "r_acceleration": 0.05,
                
                "sh_tau": 1.2,
                "sh_n": 4,
                "cbf_alpha": 3.0,

                "nlopt_maxeval": 120,
                "nlopt_xtol_rel": 1e-3,
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            rviz_arg,
            sim_node,
            obstacles_node,
            controller_node,
            rviz_node,
        ]
    )