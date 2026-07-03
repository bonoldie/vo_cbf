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
import time

from jax import config
config.update("jax_enable_x64", True)

from .core.sh_cbf import *
import jax.numpy as jnp




class CBFControllerNode(Node):
    def __init__(self):
        super().__init__("cbf_controller_node")

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

        self.declare_parameter("cruising_speed", 0.3)

        self.declare_parameter("q_terminal", 25.0)
        self.declare_parameter("q_position",  0.5)
        self.declare_parameter("q_heading",  0.5)
        self.declare_parameter("q_cruising_speed", 0.5)

        self.declare_parameter("r_acceleration", 0.05)

        self.declare_parameter("sh_tau", 1.2)
        self.declare_parameter("sh_n", 4)
        self.declare_parameter("cbf_alpha", 3.0)


        self.declare_parameter("nlopt_maxeval", 120)
        self.declare_parameter("nlopt_xtol_rel", 1e-3)

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

        # MPC parameters
        self.mpc_horizon = int(self.get_parameter("mpc_horizon").value)
        self.mpc_dt = float(self.get_parameter("mpc_dt").value)

        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.obstacle_radius_default = float(
            self.get_parameter("obstacle_radius_default").value
        )

        self.cruising_speed = float(self.get_parameter("cruising_speed").value)

        self.q_terminal = float(self.get_parameter("q_terminal").value)
        self.q_position = float(self.get_parameter("q_position").value)
        self.q_heading = float(self.get_parameter("q_heading").value)
        self.q_cruising_speed = float(self.get_parameter("q_cruising_speed").value)

        self.r_acceleration = float(self.get_parameter("r_acceleration").value)

        self.sh_tau = float(self.get_parameter("sh_tau").value)
        self.sh_n = int(self.get_parameter("sh_n").value)
        self.cbf_alpha = float(self.get_parameter("cbf_alpha").value)

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
    

    def nlopt_jax_objective(self, u, grad, robot_state0, target_position):
        objective_params = np.array(
            [
                self.mpc_dt,
                self.q_position,
                self.q_terminal,
                self.q_cruising_speed,
                self.cruising_speed,
                self.q_heading,
                self.r_acceleration,
            ],
            dtype=np.float64,
        )

        if grad.size > 0:
            grad[:] = np.asarray(
                jax_mpc_objective_grad(
                    jnp.asarray(u),
                    jnp.asarray(robot_state0),
                    jnp.asarray(target_position),
                    jnp.asarray(objective_params),
                )
            )

        value = jax_mpc_objective(
            jnp.asarray(u),
            jnp.asarray(robot_state0),
            jnp.asarray(target_position),
            jnp.asarray(objective_params),
        )

        return float(np.asarray(value))


    def nlopt_jax_shvo_cbf_constraint(self, u, grad, robot_state0, obstacle_state0, obstacle_radius, k):
        cbf_params = np.array(
            [
                self.mpc_dt,
                self.robot_radius,
                obstacle_radius,
                0,
                self.sh_tau,
                self.cbf_alpha,
            ],
            dtype=np.float64,
        )

        if grad.size > 0:
            grad[:] = np.asarray(
                jax_shvo_cbf_constraint_u_grad(
                    jnp.asarray(u),
                    jnp.asarray(robot_state0),
                    jnp.asarray(obstacle_state0),
                    jnp.asarray(cbf_params),
                    k=k,
                    n=self.sh_n,
                )
            )

        value = jax_shvo_cbf_constraint_u(
            jnp.asarray(u),
            jnp.asarray(robot_state0),
            jnp.asarray(obstacle_state0),
            jnp.asarray(cbf_params),
            k=k,
            n=self.sh_n,
        )

        return float(np.asarray(value))

    def mpc_objective(self, u, grad, robot, obstacles, target):
        """
        Objective minimized by NLopt.

        Simple MPC-like cost:
            - running position cost
            - terminal position cost
            - cruising speed cost
            - heading-to-target cost
            - small acceleration regularization
        """

        predicted_states = self.rollout_dynamics(robot, u)

        tx, ty = target["position"]

        cost = 0.0

        eps = 1e-6

        for k, state in enumerate(predicted_states):
            px = state["position"][0]
            py = state["position"][1]
            vx = state["velocity"][0]
            vy = state["velocity"][1]

            ax = u[2 * k]
            ay = u[2 * k + 1]

            # -------------------------------------------------
            # Position error
            # -------------------------------------------------
            ex = tx - px
            ey = ty - py

            dist_to_target_sq = ex * ex + ey * ey
            dist_to_target = math.sqrt(dist_to_target_sq + eps)

            # Running position cost
            cost += self.q_position * dist_to_target_sq

            # Terminal position cost
            if k == len(predicted_states) - 1:
                cost += self.q_terminal * dist_to_target_sq

            # -------------------------------------------------
            # Cruising speed cost
            #
            # Penalize deviation from desired speed:
            #     (||v|| - v_ref)^2
            # -------------------------------------------------
            speed = math.sqrt(vx * vx + vy * vy + eps)

            speed_error = speed - self.cruising_speed
            cost += self.q_cruising_speed * speed_error * speed_error

            # -------------------------------------------------
            # Heading cost
            #
            # Desired direction is from robot to target.
            # Current direction is velocity direction.
            #
            # cos(theta) = v_hat dot target_hat
            #
            # Cost:
            #     1 - cos(theta)
            #
            # This is 0 when velocity points to target,
            # and large when it points away.
            # -------------------------------------------------
            target_dir_x = ex / dist_to_target
            target_dir_y = ey / dist_to_target

            vel_dir_x = vx / speed
            vel_dir_y = vy / speed

            heading_alignment = (
                vel_dir_x * target_dir_x
                + vel_dir_y * target_dir_y
            )

            heading_cost = 1.0 - heading_alignment

            cost += self.q_heading * heading_cost

            # -------------------------------------------------
            # Small acceleration regularization
            #
            # Keeps NLopt from choosing unnecessarily aggressive
            # acceleration if many commands have similar tracking cost.
            # -------------------------------------------------
            accel_sq = ax * ax + ay * ay
            cost += self.r_acceleration * accel_sq

        return float(cost)

    def compute_command(self, robot, obstacles, target):
        px, py = robot["position"]
        vx, vy = robot["velocity"]

        tx, ty = target["position"]

        N = self.mpc_horizon
        n_vars = 2 * N

        robot_state0 = np.array(
            [
                px,
                py,
                vx,
                vy,
            ],
            dtype=np.float64,
        )

        target_position = np.array(
            [
                tx,
                ty,
            ],
            dtype=np.float64,
        )

        # Initial guess
        u0 = np.zeros(n_vars, dtype=np.float64)

        if self.previous_solution is not None and len(self.previous_solution) == n_vars:
            u0 = np.roll(self.previous_solution, -2)

        u0 = np.clip(u0, -self.max_accel, self.max_accel)

        opt = nlopt.opt(nlopt.LD_SLSQP, n_vars)

        opt.set_lower_bounds(-self.max_accel * np.ones(n_vars))
        opt.set_upper_bounds(self.max_accel * np.ones(n_vars))

        opt.set_min_objective(
            lambda u, grad: self.nlopt_jax_objective(
                u,
                grad,
                robot_state0,
                target_position,
            )
        )

        # SH-VO CBF constraints
        for k in range(N):
            for obs in obstacles:
                ox, oy = obs["position"]
                ovx, ovy = obs["velocity"]

                obstacle_state0 = np.array(
                    [
                        ox,
                        oy,
                        ovx,
                        ovy,
                    ],
                    dtype=np.float64,
                )

                obstacle_radius = float(
                    obs.get("radius", self.obstacle_radius_default)
                )

                opt.add_inequality_constraint(
                    lambda u, grad, kk=k, oo=obstacle_state0, rr=obstacle_radius: self.nlopt_jax_shvo_cbf_constraint(
                        u,
                        grad,
                        robot_state0,
                        oo,
                        rr,
                        kk,
                    ),
                    1e-6,
                )

        opt.set_xtol_rel(self.nlopt_xtol_rel)
        opt.set_maxeval(self.nlopt_maxeval)

        u_star = opt.optimize(u0)

        result_code = opt.last_optimize_result()
        objective_value = opt.last_optimum_value()

        self.previous_solution = np.array(u_star)

        ax = float(u_star[0])
        ay = float(u_star[1])

        ax, ay = self.limit_acceleration(ax, ay)

        self.get_logger().info(
            f"NLopt result={result_code} | "
            f"cost={objective_value:.3f} | "
            f"cmd=({ax:.3f}, {ay:.3f}) | "
            f"obs={len(obstacles)}",
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

    node = CBFControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()