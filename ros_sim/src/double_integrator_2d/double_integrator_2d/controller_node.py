#!/usr/bin/env python3

import math
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Twist, PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_ros import TransformException


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("robot_frame_id", "base_link")
        self.declare_parameter("target_frame_id", "target")

        self.declare_parameter(
            "obstacle_frames",
            [
                "obs_fixed_1",
                "obs_fixed_2",
                "obs_circle_1",
                "obs_line_1",
            ],
        )

        self.declare_parameter("dt", 0.02)

        self.declare_parameter("target_x", 4.0)
        self.declare_parameter("target_y", 0.0)

        self.declare_parameter("max_accel", 2.0)
        self.declare_parameter("target_tolerance", 0.05)

        # Simple PD stub gains
        self.declare_parameter("kp_position", 1.5)
        self.declare_parameter("kd_velocity", 1.2)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.robot_frame_id = str(self.get_parameter("robot_frame_id").value)
        self.target_frame_id = str(self.get_parameter("target_frame_id").value)

        self.obstacle_frames = list(
            self.get_parameter("obstacle_frames").value
        )

        self.dt = float(self.get_parameter("dt").value)

        self.target_x = float(self.get_parameter("target_x").value)
        self.target_y = float(self.get_parameter("target_y").value)

        self.max_accel = float(self.get_parameter("max_accel").value)
        self.target_tolerance = float(
            self.get_parameter("target_tolerance").value
        )

        self.kp_position = float(self.get_parameter("kp_position").value)
        self.kd_velocity = float(self.get_parameter("kd_velocity").value)

        # -------------------------------------------------
        # Robot state
        # -------------------------------------------------
        self.robot_state = {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "yaw": 0.0,
            "stamp": None,
        }

        self.have_odom = False

        # -------------------------------------------------
        # Obstacle velocity estimation memory
        # -------------------------------------------------
        self.previous_obstacle_positions: Dict[str, Tuple[float, float]] = {}
        self.previous_obstacle_times: Dict[str, float] = {}

        # -------------------------------------------------
        # TF
        # -------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # -------------------------------------------------
        # ROS interfaces
        # -------------------------------------------------
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        # RViz "Publish Point" tool publishes here by default.
        # Use it to move the target by clicking in RViz.
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            "/clicked_point",
            self.clicked_point_callback,
            10,
        )

        # Optional explicit target topic.
        self.target_sub = self.create_subscription(
            PointStamped,
            "/target_position",
            self.target_position_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_accel",
            10,
        )

        self.target_marker_pub = self.create_publisher(
            Marker,
            "/target_marker",
            10,
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Controller node started.")
        self.get_logger().info("Reads robot from /odom.")
        self.get_logger().info("Reads obstacles from TF.")
        self.get_logger().info("Publishes acceleration to /cmd_accel.")
        self.get_logger().info("Publishes target marker to /target_marker.")
        self.get_logger().info("Use RViz Publish Point tool to update target.")

    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------

    def odom_callback(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.z, q.w)

        self.robot_state = {
            "position": [px, py],
            "velocity": [vx, vy],
            "yaw": yaw,
            "stamp": msg.header.stamp,
        }

        self.have_odom = True

    def clicked_point_callback(self, msg: PointStamped):
        self.set_target_from_point(msg)

    def target_position_callback(self, msg: PointStamped):
        self.set_target_from_point(msg)

    def set_target_from_point(self, msg: PointStamped):
        if msg.header.frame_id != self.frame_id:
            self.get_logger().warn(
                f"Target point frame is '{msg.header.frame_id}', "
                f"but expected '{self.frame_id}'. "
                "For now I am using the raw coordinates."
            )

        self.target_x = msg.point.x
        self.target_y = msg.point.y

        self.get_logger().info(
            f"New target: x={self.target_x:.3f}, y={self.target_y:.3f}"
        )

    # -------------------------------------------------
    # Main control loop
    # -------------------------------------------------

    def control_loop(self):
        now = self.get_clock().now()

        self.publish_target_marker(now)
        self.publish_target_tf(now)

        if not self.have_odom:
            self.publish_zero_command()
            return

        obstacles = self.read_obstacles_from_tf(now)

        target = {
            "position": [self.target_x, self.target_y],
            "frame_id": self.frame_id,
        }

        ax, ay = self.compute_command(
            robot=self.robot_state,
            obstacles=obstacles,
            target=target,
        )

        self.publish_acceleration_command(ax, ay)

    # -------------------------------------------------
    # Obstacle reading
    # -------------------------------------------------

    def read_obstacles_from_tf(self, now) -> List[dict]:
        obstacles = []

        now_sec = now.nanoseconds * 1e-9

        for frame in self.obstacle_frames:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    frame,
                    Time(),
                )

                x = tf_msg.transform.translation.x
                y = tf_msg.transform.translation.y

                vx, vy = self.estimate_obstacle_velocity(
                    name=frame,
                    x=x,
                    y=y,
                    now_sec=now_sec,
                )

                obstacles.append(
                    {
                        "name": frame,
                        "position": [x, y],
                        "velocity": [vx, vy],
                        "frame_id": frame,
                    }
                )

            except TransformException as ex:
                self.get_logger().warn(
                    f"Could not read TF {self.frame_id} -> {frame}: {ex}",
                    throttle_duration_sec=1.0,
                )

        return obstacles

    def estimate_obstacle_velocity(self, name, x, y, now_sec):
        if name not in self.previous_obstacle_positions:
            self.previous_obstacle_positions[name] = (x, y)
            self.previous_obstacle_times[name] = now_sec
            return 0.0, 0.0

        px, py = self.previous_obstacle_positions[name]
        pt = self.previous_obstacle_times[name]

        dt = now_sec - pt

        if dt <= 1e-6:
            return 0.0, 0.0

        vx = (x - px) / dt
        vy = (y - py) / dt

        self.previous_obstacle_positions[name] = (x, y)
        self.previous_obstacle_times[name] = now_sec

        return vx, vy

    # -------------------------------------------------
    # Control stub
    # -------------------------------------------------

    def compute_command(self, robot, obstacles, target):
        """
        Stub controller.

        Inputs:
            robot:
                {
                    "position": [px, py],
                    "velocity": [vx, vy],
                    "yaw": yaw,
                    "stamp": ...
                }

            obstacles:
                [
                    {
                        "name": "obs_circle_1",
                        "position": [ox, oy],
                        "velocity": [ovx, ovy],
                        "frame_id": "obs_circle_1"
                    },
                    ...
                ]

            target:
                {
                    "position": [tx, ty],
                    "frame_id": "odom"
                }

        Output:
            ax, ay

        Replace this function with MPC, CBF-QP, VO, MPPI, etc.
        """

        px, py = robot["position"]
        vx, vy = robot["velocity"]

        tx, ty = target["position"]

        ex = tx - px
        ey = ty - py

        distance_to_target = math.hypot(ex, ey)
        speed = math.hypot(vx, vy)

        # Stop when close enough.
        if distance_to_target < self.target_tolerance and speed < 0.05:
            return 0.0, 0.0

        # -------------------------------------------------
        # Simple PD point stabilizer:
        #
        # a = kp * (p_target - p_robot) - kd * v_robot
        #
        # This ignores obstacles for now.
        # Obstacles are already passed here so you can replace
        # this block with CBF/VO/MPC constraints.
        # -------------------------------------------------

        ax = self.kp_position * ex - self.kd_velocity * vx
        ay = self.kp_position * ey - self.kd_velocity * vy

        # Example place-holder for future obstacle logic:
        #
        # for obs in obstacles:
        #     ox, oy = obs["position"]
        #     ovx, ovy = obs["velocity"]
        #
        #     relative_position = [ox - px, oy - py]
        #     relative_velocity = [ovx - vx, ovy - vy]
        #
        #     Use these for:
        #       - velocity obstacles
        #       - CBF constraints
        #       - MPC obstacle constraints
        #       - MPPI costs

        ax, ay = self.limit_acceleration(ax, ay)

        return ax, ay

    # -------------------------------------------------
    # Publishers
    # -------------------------------------------------

    def publish_acceleration_command(self, ax, ay):
        cmd = Twist()

        cmd.linear.x = ax
        cmd.linear.y = ay
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def publish_zero_command(self):
        self.publish_acceleration_command(0.0, 0.0)

    def publish_target_marker(self, stamp):
        marker = Marker()

        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = self.frame_id

        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = self.target_x
        marker.pose.position.y = self.target_y
        marker.pose.position.z = 0.08
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.16
        marker.scale.y = 0.16
        marker.scale.z = 0.16

        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 1.0

        self.target_marker_pub.publish(marker)

    def publish_target_tf(self, stamp):
        tf_msg = TransformStamped()

        tf_msg.header.stamp = stamp.to_msg()
        tf_msg.header.frame_id = self.frame_id
        tf_msg.child_frame_id = self.target_frame_id

        tf_msg.transform.translation.x = self.target_x
        tf_msg.transform.translation.y = self.target_y
        tf_msg.transform.translation.z = 0.0

        tf_msg.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(tf_msg)

    # -------------------------------------------------
    # Utils
    # -------------------------------------------------

    def limit_acceleration(self, ax, ay):
        norm = math.hypot(ax, ay)

        if norm <= self.max_accel:
            return ax, ay

        scale = self.max_accel / norm

        return ax * scale, ay * scale

    @staticmethod
    def quaternion_to_yaw(qz, qw):
        return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


def main(args=None):
    rclpy.init(args=args)

    node = ControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()