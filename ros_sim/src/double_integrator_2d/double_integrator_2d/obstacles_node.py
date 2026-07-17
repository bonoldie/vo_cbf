#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

from interfaces.srv import GetObstacles
from interfaces.msg import Obstacle


class ObstaclesNode(Node):
    def __init__(self):
        super().__init__("obstacles_node")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("path_history_size", 500)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.dt = float(self.get_parameter("dt").value)
        self.path_history_size = int(self.get_parameter("path_history_size").value)

        # ----------------------------
        # Obstacle definitions
        # ----------------------------
        # Fixed obstacles:
        #
        #   position is constant.
        #
        # Moving obstacles:
        #
        #   trajectory = "circle"
        #   p(t) = center + radius * [cos(wt + phase), sin(wt + phase)]
        #
        #   trajectory = "line"
        #   p(t) = p0 + direction * amplitude * sin(wt + phase)
        #
        # Every obstacle publishes:
        #   - a marker
        #   - a TF frame: odom -> obstacle_name
        # ----------------------------

        self.obstacles = [
            # {
            #     "name": "obs_line_1",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [0.0, 1.0],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 1.5707963268,
            # },
            # {
            #     "name": "obs_line_2",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-0.3090169944, 0.9510565163],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 2.1991148575,
            # },
            # {
            #     "name": "obs_line_3",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-0.5877852523, 0.8090169944],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 2.8274333882,
            # },
            # {
            #     "name": "obs_line_4",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-0.8090169944, 0.5877852523],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 3.4557519189,
            # },
            # {
            #     "name": "obs_line_5",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-0.9510565163, 0.3090169944],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 4.0840704497,
            # },
            # {
            #     "name": "obs_line_6",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-1.0, 0.0],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 4.7123889804,
            # },
            # {
            #     "name": "obs_line_7",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-0.9510565163, -0.3090169944],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 5.3407075111,
            # },
            # {
            #     "name": "obs_line_8",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-0.8090169944, -0.5877852523],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 5.9690260418,
            # },
            # {
            #     "name": "obs_line_9",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-0.5877852523, -0.8090169944],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 0.3141592654,
            # },
            # {
            #     "name": "obs_line_10",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [2.0, 0.0],
            #     "direction": [-0.3090169944, -0.9510565163],
            #     "amplitude": 2.0,
            #     "omega": 0.1,
            #     "phase": 0.9424777961,
            # }
            {
                "name": "obs_fixed_2",
                "kind": "fixed",
                "radius": 0.25,
                "position": [2.0, -0.8],
            }
            # {
            #     "name": "obs_fixed_3",
            #     "kind": "fixed",
            #     "radius": 0.25,
            #     "position": [3.0, 0.1],
            # },
            # {
            #     "name": "obs_fixed_1",
            #     "kind": "fixed",
            #     "radius": 0.25,
            #     "position": [2.0, 0.8],
            # },
            # {
            #     "name": "obs_circle_1",
            #     "kind": "moving",
            #     "trajectory": "circle",
            #     "radius": 0.25,
            #     "center": [1.2, 1.4],
            #     "traj_radius": 0.6,
            #     "omega": 0.06,
            #     "phase": 0.0,
            # },
            # {
            #     "name": "obs_line_1",
            #     "kind": "moving",
            #     "trajectory": "line",
            #     "radius": 0.25,
            #     "p0": [0.0, 0.0],
            #     "direction": [0.0, 1.0],
            #     "amplitude": 1.0,
            #     "omega": 0.1,
            #     "phase": 0.0,
            # }
        ]

        self.start_time = self.get_clock().now()
        self.histories: Dict[str, List[Tuple[float, float]]] = {
            obs["name"]: [] for obs in self.obstacles
        }

        # ----------------------------
        # ROS interfaces
        # ----------------------------
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/obstacle_markers",
            10,
        )

        self.service = self.create_service(
            GetObstacles,
            "get_obstacles",
            self.get_obstacles_callback,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info("Obstacles node started.")
        self.get_logger().info("Publishing /obstacle_markers and obstacle TF frames.")
        self.get_logger().info("A list of the obstacles is available by calling the /get_obstacles service.")


    def get_obstacles_callback(self, request: GetObstacles.Request, response: GetObstacles.Response) -> GetObstacles.Response:

        for obs in self.obstacles:
            msg = Obstacle()
            coords = (
                obs.get("position")
                or obs.get("center")
                or obs.get("p0")
            )

            msg.name = obs["name"]
            msg.dimensions = len(coords) if coords is not None else 0
            msg.kind = obs["kind"]
            msg.radius = obs["radius"]

            response.obstacles.append(msg)
        return response


    def update(self):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9

        marker_array = MarkerArray()

        for i, obs in enumerate(self.obstacles):
            x, y, vx, vy = self.evaluate_obstacle(obs, t)

            name = obs["name"]
            radius = float(obs["radius"])

            self.publish_obstacle_tf(now, name, x, y)

            self.histories[name].append((x, y))
            if len(self.histories[name]) > self.path_history_size:
                self.histories[name].pop(0)

            sphere_marker = self.make_sphere_marker(
                stamp=now,
                marker_id=10 * i,
                name=name,
                x=x,
                y=y,
                radius=radius,
                moving=(obs["kind"] == "moving"),
            )

            text_marker = self.make_text_marker(
                stamp=now,
                marker_id=10 * i + 1,
                name=name,
                x=x,
                y=y,
                radius=radius,
            )

            path_marker = self.make_path_marker(
                stamp=now,
                marker_id=10 * i + 2,
                name=name,
                history=self.histories[name],
                moving=(obs["kind"] == "moving"),
            )

            velocity_marker = self.make_velocity_marker(
                stamp=now,
                marker_id=10 * i + 3,
                name=name,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
            )

            marker_array.markers.append(sphere_marker)
            marker_array.markers.append(text_marker)
            marker_array.markers.append(path_marker)
            marker_array.markers.append(velocity_marker)

        self.marker_pub.publish(marker_array)

    def evaluate_obstacle(self, obs, t):
        if obs["kind"] == "fixed":
            x = float(obs["position"][0])
            y = float(obs["position"][1])
            vx = 0.0
            vy = 0.0
            return x, y, vx, vy

        trajectory = obs["trajectory"]

        if trajectory == "circle":
            cx = float(obs["center"][0])
            cy = float(obs["center"][1])
            r = float(obs["traj_radius"])
            w = float(obs["omega"])
            phase = float(obs.get("phase", 0.0))

            angle = w * t + phase

            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)

            vx = -r * w * math.sin(angle)
            vy = r * w * math.cos(angle)

            return x, y, vx, vy

        if trajectory == "line":
            p0x = float(obs["p0"][0])
            p0y = float(obs["p0"][1])

            dx = float(obs["direction"][0])
            dy = float(obs["direction"][1])

            norm = math.hypot(dx, dy)
            if norm < 1e-9:
                dx = 1.0
                dy = 0.0
            else:
                dx /= norm
                dy /= norm

            amp = float(obs["amplitude"])
            w = float(obs["omega"])
            phase = float(obs.get("phase", 0.0))

            s = amp * math.sin(w * t + phase)
            s_dot = amp * w * math.cos(w * t + phase)

            x = p0x + dx * s
            y = p0y + dy * s

            vx = dx * s_dot
            vy = dy * s_dot

            return x, y, vx, vy

        raise ValueError(f"Unknown obstacle trajectory: {trajectory}")

    def publish_obstacle_tf(self, stamp, name, x, y):
        tf_msg = TransformStamped()

        tf_msg.header.stamp = stamp.to_msg()
        tf_msg.header.frame_id = self.frame_id
        tf_msg.child_frame_id = name

        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = 0.0

        tf_msg.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(tf_msg)

    def make_sphere_marker(self, stamp, marker_id, name, x, y, radius, moving):
        marker = Marker()

        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = self.frame_id

        marker.ns = "obstacles_spheres"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = radius
        marker.pose.orientation.w = 1.0

        marker.scale.x = 2.0 * radius
        marker.scale.y = 2.0 * radius
        marker.scale.z = 2.0 * radius

        if moving:
            marker.color.r = 1.0
            marker.color.g = 0.35
            marker.color.b = 0.1
            marker.color.a = 0.9
        else:
            marker.color.r = 0.8
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.color.a = 0.9

        return marker

    def make_text_marker(self, stamp, marker_id, name, x, y, radius):
        marker = Marker()

        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = self.frame_id

        marker.ns = "obstacle_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 2.0 * radius + 0.15
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.18

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = name

        return marker

    def make_path_marker(self, stamp, marker_id, name, history, moving):
        marker = Marker()

        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = self.frame_id

        marker.ns = "obstacle_paths"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.03

        if moving:
            marker.color.r = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.1
            marker.color.a = 0.8
        else:
            marker.color.r = 0.4
            marker.color.g = 0.4
            marker.color.b = 0.4
            marker.color.a = 0.2

        marker.points = []

        for x, y in history:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.03
            marker.points.append(p)

        return marker

    def make_velocity_marker(self, stamp, marker_id, name, x, y, vx, vy):
        marker = Marker()

        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = self.frame_id

        marker.ns = "obstacle_velocities"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        speed = math.hypot(vx, vy)

        marker.points = []

        p0 = Point()
        p0.x = x
        p0.y = y
        p0.z = 0.10

        p1 = Point()
        p1.x = x + vx
        p1.y = y + vy
        p1.z = 0.10

        marker.points.append(p0)
        marker.points.append(p1)

        marker.scale.x = 0.035
        marker.scale.y = 0.08
        marker.scale.z = 0.08

        marker.color.r = 0.2
        marker.color.g = 1.0
        marker.color.b = 0.2
        marker.color.a = 0.9 if speed > 1e-6 else 0.0

        return marker


def main(args=None):
    rclpy.init(args=args)

    node = ObstaclesNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()