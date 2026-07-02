#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster


class DoubleIntegrator2DSim(Node):
    def __init__(self):
        super().__init__("double_integrator_2d_sim")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("dt", 0.01)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)

        self.declare_parameter("robot_radius", 0.15)
        self.declare_parameter("max_accel", 2.0)
        self.declare_parameter("max_velocity", 5.0)
        self.declare_parameter("cmd_timeout", 0.5)

        self.dt = float(self.get_parameter("dt").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.child_frame_id = str(self.get_parameter("child_frame_id").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.max_accel = float(self.get_parameter("max_accel").value)
        self.max_velocity = float(self.get_parameter("max_velocity").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)

        # ----------------------------
        # State: [x, y, vx, vy]
        # ----------------------------
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.last_yaw = 0.0

        # Input: [ax, ay]
        self.ax_cmd = 0.0
        self.ay_cmd = 0.0
        self.last_cmd_time = self.get_clock().now()

        # ----------------------------
        # ROS interfaces
        # ----------------------------
        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_accel",
            self.cmd_callback,
            10,
        )

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.path_pub = self.create_publisher(Path, "/path", 10)
        self.marker_pub = self.create_publisher(Marker, "/robot_marker", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.path = Path()
        self.path.header.frame_id = self.frame_id

        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info("2D double-integrator simulator started.")
        self.get_logger().info("Input topic: /cmd_accel")
        self.get_logger().info("Use Twist.linear.x = ax, Twist.linear.y = ay")

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min(value, max_value), min_value)

    @staticmethod
    def yaw_to_quaternion(yaw):
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qz, qw

    def cmd_callback(self, msg: Twist):
        self.ax_cmd = self.clamp(
            msg.linear.x,
            -self.max_accel,
            self.max_accel,
        )
        self.ay_cmd = self.clamp(
            msg.linear.y,
            -self.max_accel,
            self.max_accel,
        )

        self.last_cmd_time = self.get_clock().now()

    def update(self):
        now = self.get_clock().now()

        elapsed_since_cmd = (now - self.last_cmd_time).nanoseconds * 1e-9

        if elapsed_since_cmd > self.cmd_timeout:
            ax = 0.0
            ay = 0.0
        else:
            ax = self.ax_cmd
            ay = self.ay_cmd

        # -------------------------------------------------
        # Double-integrator dynamics
        #
        # px_dot = vx
        # py_dot = vy
        # vx_dot = ax
        # vy_dot = ay
        #
        # Semi-implicit Euler:
        # v[k+1] = v[k] + a[k] dt
        # p[k+1] = p[k] + v[k+1] dt
        # -------------------------------------------------

        self.vx += ax * self.dt
        self.vy += ay * self.dt

        self.vx = self.clamp(self.vx, -self.max_velocity, self.max_velocity)
        self.vy = self.clamp(self.vy, -self.max_velocity, self.max_velocity)

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        speed = math.hypot(self.vx, self.vy)
        if speed > 1e-6:
            self.last_yaw = math.atan2(self.vy, self.vx)

        yaw = self.last_yaw

        self.publish_odom(now, yaw, ax, ay)
        self.publish_path(now, yaw)
        self.publish_robot_marker(now)

        if self.publish_tf:
            self.publish_transform(now, yaw)

    def publish_odom(self, stamp, yaw, ax, ay):
        odom = Odometry()

        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        qz, qw = self.yaw_to_quaternion(yaw)
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom)

    def publish_path(self, stamp, yaw):
        pose = PoseStamped()

        pose.header.stamp = stamp.to_msg()
        pose.header.frame_id = self.frame_id

        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = 0.0

        qz, qw = self.yaw_to_quaternion(yaw)
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.path.header.stamp = stamp.to_msg()
        self.path.poses.append(pose)

        if len(self.path.poses) > 4000:
            self.path.poses.pop(0)

        self.path_pub.publish(self.path)

    def publish_transform(self, stamp, yaw):
        tf_msg = TransformStamped()

        tf_msg.header.stamp = stamp.to_msg()
        tf_msg.header.frame_id = self.frame_id
        tf_msg.child_frame_id = self.child_frame_id

        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0

        qz, qw = self.yaw_to_quaternion(yaw)
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf_msg)

    def publish_robot_marker(self, stamp):
        marker = Marker()

        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = self.child_frame_id

        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Marker is attached to base_link.
        # z = radius means the sphere touches the z = 0 plane.
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = self.robot_radius

        marker.pose.orientation.w = 1.0

        marker.scale.x = 2.0 * self.robot_radius
        marker.scale.y = 2.0 * self.robot_radius
        marker.scale.z = 2.0 * self.robot_radius

        marker.color.r = 0.1
        marker.color.g = 0.4
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.frame_locked = True

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)

    node = DoubleIntegrator2DSim()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()