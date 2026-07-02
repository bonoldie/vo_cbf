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

import numpy as np
import nlopt


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

        self.declare_parameter("max_accel", 0.5)
        self.declare_parameter("target_tolerance", 0.005)

        # -------------------------------------------------
        # MPC / NLopt parameters
        # -------------------------------------------------
        self.declare_parameter("mpc_horizon", 10)
        self.declare_parameter("mpc_dt", 0.10)

        self.declare_parameter("robot_radius", 0.15)
        self.declare_parameter("obstacle_radius_default", 0.25)
        self.declare_parameter("safety_margin", 0.15)

        self.declare_parameter("q_position", 8.0)
        self.declare_parameter("q_terminal", 25.0)
        self.declare_parameter("q_velocity", 0.3)
        self.declare_parameter("r_acceleration", 0.08)
        self.declare_parameter("r_acceleration_smooth", 0.15)
        self.declare_parameter("q_obstacle_soft", 10.0)

        self.declare_parameter("nlopt_maxeval", 120)
        self.declare_parameter("nlopt_xtol_rel", 1e-3)

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

        # MPC parameters
        self.mpc_horizon = int(self.get_parameter("mpc_horizon").value)
        self.mpc_dt = float(self.get_parameter("mpc_dt").value)

        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.obstacle_radius_default = float(
            self.get_parameter("obstacle_radius_default").value
        )
        self.safety_margin = float(self.get_parameter("safety_margin").value)

        self.q_position = float(self.get_parameter("q_position").value)
        self.q_terminal = float(self.get_parameter("q_terminal").value)
        self.q_velocity = float(self.get_parameter("q_velocity").value)
        self.r_acceleration = float(self.get_parameter("r_acceleration").value)
        self.r_acceleration_smooth = float(
            self.get_parameter("r_acceleration_smooth").value
        )
        self.q_obstacle_soft = float(self.get_parameter("q_obstacle_soft").value)

        self.nlopt_maxeval = int(self.get_parameter("nlopt_maxeval").value)
        self.nlopt_xtol_rel = float(self.get_parameter("nlopt_xtol_rel").value)

        self.previous_solution = None

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

    def obstacle_constraint(self, u, grad, robot, obstacle, k):
        """
        Hard nonlinear obstacle constraint for one obstacle at one horizon step.

        NLopt requires:
            g(u) <= 0

        We impose:
            distance(robot_k, obstacle_k) >= safety_radius

        Therefore:
            safety_radius^2 - distance^2 <= 0
        """

        predicted_states = self.rollout_dynamics(robot, u)

        robot_state_k = predicted_states[k]

        px = robot_state_k["position"][0]
        py = robot_state_k["position"][1]

        ox, oy = self.predict_obstacle_position(obstacle, k)

        dx = px - ox
        dy = py - oy

        distance_sq = dx * dx + dy * dy

        safety_radius = (
            self.robot_radius
            + self.obstacle_radius_default
            + self.safety_margin
        )

        return safety_radius * safety_radius - distance_sq

    def predict_obstacle_position(self, obstacle, k):
        """
        Predict obstacle position k steps into the horizon.

        Since the controller reads obstacle position and velocity from TF,
        this uses a constant-velocity prediction.
        """

        ox, oy = obstacle["position"]
        ovx, ovy = obstacle["velocity"]

        t = (k + 1) * self.mpc_dt

        pred_x = ox + ovx * t
        pred_y = oy + ovy * t

        return pred_x, pred_y

    def rollout_dynamics(self, robot, u):
        """
        Predict robot states over the MPC horizon using the same 2D double-integrator model.
        """

        px, py = robot["position"]
        vx, vy = robot["velocity"]

        N = self.mpc_horizon
        dt = self.mpc_dt

        states = []

        for k in range(N):
            ax = float(u[2 * k])
            ay = float(u[2 * k + 1])

            vx = vx + ax * dt
            vy = vy + ay * dt

            px = px + vx * dt
            py = py + vy * dt

            states.append(
                {
                    "position": [px, py],
                    "velocity": [vx, vy],
                }
            )

        return states

    def mpc_objective(self, u, grad, robot, obstacles, target):
        """
        Objective minimized by NLopt.

        This is derivative-free because we use LN_COBYLA.
        Therefore grad is ignored.
        """

        predicted_states = self.rollout_dynamics(robot, u)

        tx, ty = target["position"]

        cost = 0.0

        previous_ax = 0.0
        previous_ay = 0.0

        for k, state in enumerate(predicted_states):
            px = state["position"][0]
            py = state["position"][1]
            vx = state["velocity"][0]
            vy = state["velocity"][1]

            ax = u[2 * k]
            ay = u[2 * k + 1]

            ex = tx - px
            ey = ty - py

            dist_to_target_sq = ex * ex + ey * ey
            speed_sq = vx * vx + vy * vy
            accel_sq = ax * ax + ay * ay

            if k == len(predicted_states) - 1:
                cost += self.q_terminal * dist_to_target_sq
            else:
                cost += self.q_position * dist_to_target_sq

            cost += self.q_velocity * speed_sq
            cost += self.r_acceleration * accel_sq

            dax = ax - previous_ax
            day = ay - previous_ay

            cost += self.r_acceleration_smooth * (dax * dax + day * day)

            previous_ax = ax
            previous_ay = ay

            # Soft obstacle penalty.
            # The hard constraints already enforce clearance if feasible.
            # This term helps the optimizer prefer larger clearance.
            for obs in obstacles:
                ox, oy = self.predict_obstacle_position(obs, k)

                dx = px - ox
                dy = py - oy

                dist = math.hypot(dx, dy)

                safety_radius = (
                    self.robot_radius
                    + self.obstacle_radius_default
                    + self.safety_margin
                )

                influence_radius = safety_radius + 0.8

                if dist < influence_radius:
                    violation = influence_radius - dist
                    cost += self.q_obstacle_soft * violation * violation

        return float(cost)

    def compute_command(self, robot, obstacles, target):
        """
        MPC-like obstacle-avoidance controller using NLopt.

        Decision variable:
            u = [ax0, ay0, ax1, ay1, ..., axN-1, ayN-1]

        It optimizes an acceleration sequence and applies only the first input.
        """

        px, py = robot["position"]
        vx, vy = robot["velocity"]

        tx, ty = target["position"]

        N = self.mpc_horizon
        dt = self.mpc_dt
        n_vars = 2 * N

        # -------------------------------------------------
        # Initial guess
        # -------------------------------------------------
        if self.previous_solution is not None and len(self.previous_solution) == n_vars:
            u0 = np.roll(self.previous_solution, -2)
            u0[-2:] = 0.0
        else:
            # Start from simple PD acceleration repeated over the horizon.
            ex = tx - px
            ey = ty - py

            ax0 = self.kp_position * ex - self.kd_velocity * vx
            ay0 = self.kp_position * ey - self.kd_velocity * vy

            ax0, ay0 = self.limit_acceleration(ax0, ay0)

            u0 = np.zeros(n_vars)
            for k in range(N):
                u0[2 * k] = ax0
                u0[2 * k + 1] = ay0

        u0 = np.clip(u0, -self.max_accel, self.max_accel)

        # -------------------------------------------------
        # NLopt optimizer
        # -------------------------------------------------
        opt = nlopt.opt(nlopt.LN_COBYLA, n_vars)

        lower_bounds = -self.max_accel * np.ones(n_vars)
        upper_bounds = self.max_accel * np.ones(n_vars)

        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)

        opt.set_min_objective(
            lambda u, grad: self.mpc_objective(
                u,
                grad,
                robot,
                obstacles,
                target,
            )
        )

        # -------------------------------------------------
        # Hard obstacle constraints
        #
        # NLopt inequality convention:
        #     g(u) <= 0
        #
        # We want:
        #     distance^2 >= safety_radius^2
        #
        # Therefore:
        #     safety_radius^2 - distance^2 <= 0
        # -------------------------------------------------
        for k in range(N):
            for obs_index, obs in enumerate(obstacles):
                opt.add_inequality_constraint(
                    lambda u, grad, kk=k, oo=obs: self.obstacle_constraint(
                        u,
                        grad,
                        robot,
                        oo,
                        kk,
                    ),
                    1e-4,
                )

        opt.set_maxeval(self.nlopt_maxeval)
        opt.set_xtol_rel(self.nlopt_xtol_rel)

        # Useful for COBYLA
        opt.set_initial_step(0.25 * np.ones(n_vars))

        # -------------------------------------------------
        # Solve
        # -------------------------------------------------
        u_star = opt.optimize(u0)

        result_code = opt.last_optimize_result()
        objective_value = opt.last_optimum_value()

        self.previous_solution = np.array(u_star)

        ax = float(u_star[0])
        ay = float(u_star[1])

        ax, ay = self.limit_acceleration(ax, ay)

        self.get_logger().info(
            f"NLopt result={result_code}, "
            f"cost={objective_value:.3f}, "
            f"cmd=({ax:.3f}, {ay:.3f}), "
            f"obstacles={len(obstacles)}",
            throttle_duration_sec=0.5,
        )

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