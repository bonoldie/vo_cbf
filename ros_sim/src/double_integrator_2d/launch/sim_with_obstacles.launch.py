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
        executable="controller",
        name="controller_node",
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

                "kp_position": 1.5,
                "kd_velocity": 1.2,

                "mpc_horizon": 10,
                "mpc_dt": 0.1,

                "robot_radius": 0.15,
                "obstacle_radius_default": 0.25,
                "safety_margin": 0.15,

                "q_position": 8.0,
                "q_terminal": 25.0,
                "q_velocity": 0.3,
                "r_acceleration": 0.08,
                "r_acceleration_smooth": 0.15,
                "q_obstacle_soft": 10.0,

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