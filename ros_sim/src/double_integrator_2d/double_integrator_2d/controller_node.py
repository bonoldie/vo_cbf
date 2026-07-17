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

from interfaces.srv import GetObstacles
from scipy.interpolate import CubicHermiteSpline
from .sh_cbf_core import *

import osqp
import scipy.sparse as sparse

class ControllerNode(Node):

    def test(self): 
        # State: [x, y, vx, vy]
        robot_state = jnp.array([0.5, 0.0, 0.0, 0.0])
        obstacle_state = jnp.array([2.0, 2.0, 0.0, 0.0])

        robot_radius = 0.2
        obstacle_radius = 0.2

        tau = 1.5
        n = 6

        def h_as_function_of_robot_state(x):
            return compute_candidate_h(
                x,
                obstacle_state,
                robot_radius,
                obstacle_radius,
                n,
                tau,
            )
                
        gradH = jax.grad(h_as_function_of_robot_state)

        # plant EQ
        u = jnp.array((10.0, 0.0))
        
        A = jnp.array(((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
        G = jnp.array(((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

        #print(f"A: {A}")
        #print(f"G: {G}")
        #print(f"gradH(robot_state): {gradH(robot_state)}")

        u_local = u # R_world_to_local @ u

        #print(f"u_local: {u_local}")
        
        h_val = class_K_function(h_as_function_of_robot_state(robot_state), gamma=1.0, beta=0)
        U_cbf = gradH(robot_state) @ A @ robot_state +  gradH(robot_state) @ G @ u_local + h_val

        # print(f"U_cbf: {U_cbf}")


    # Obstacles definition (loaded via get_obstacles srv)
    obstacles = []

    # Obstacles definition (loaded via get_obstacles srv)
    # {"obs_name": {position: [0.0, 0.0], velocity: [0.0, 0.0], at: 1233..213.123(epoch time in seconds) }}  
    obstacles_states = {}

    def load_parameters(self):

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

        # Frames
        self.declare_parameter("frame_id", "odom")
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.declare_parameter("robot_frame_id", "base_link")
        self.robot_frame_id = str(self.get_parameter("robot_frame_id").value)
        self.declare_parameter("target_frame_id", "target")
        self.target_frame_id = str(self.get_parameter("target_frame_id").value)

        # Control loop period
        self.declare_parameter("dt", 0.1)
        self.dt = float(self.get_parameter("dt").value)

        # Target and tolerances
        self.declare_parameter("target_x", 4.0)
        self.target_x = float(self.get_parameter("target_x").value)
        self.declare_parameter("target_y", 0.0)
        self.target_y = float(self.get_parameter("target_y").value)
        self.declare_parameter("target_tolerance", 0.005)
        self.target_tolerance = float(self.get_parameter("target_tolerance").value)

        # Constraints
        self.declare_parameter("max_accel", 0.5)
        self.max_accel = float(self.get_parameter("max_accel").value)
        self.declare_parameter("reference_speed", 0.5)
        self.reference_speed = float(self.get_parameter("reference_speed").value)
        
        # NLopt params
        self.declare_parameter("nlopt_maxeval", 120)
        self.nlopt_maxeval = int(self.get_parameter("nlopt_maxeval").value)
        self.declare_parameter("nlopt_xtol_rel", 1e-3)
        self.nlopt_xtol_rel = float(self.get_parameter("nlopt_xtol_rel").value)

    def load_obstacles(self): 
        req = GetObstacles.Request()
        self.future = self.create_client(GetObstacles, 'get_obstacles').call_async(req)
        rclpy.spin_until_future_complete(self, self.future)
        
        self.obstacles = {}
        for obstacle in self.future.result().obstacles:
            self.obstacles[obstacle.name] = {
                "name": obstacle.name, 
                "kind": obstacle.kind, 
                "dimensions": obstacle.dimensions,
                "radius": obstacle.radius
            }

    def __init__(self):
        super().__init__("controller_node")

        self.test()

        self.get_logger().info("Loading parameters.")
        self.load_parameters()
        self.get_logger().info("Loading obstacles.")
        self.load_obstacles()


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
                "Using the raw coordinates."
            )

        self.target_x = msg.point.x
        self.target_y = msg.point.y

        self.get_logger().info(f"New target: x={self.target_x:.3f}, y={self.target_y:.3f}")

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

        self.update_obstacles_states(now)

        target_state = {
            "position": [self.target_x, self.target_y],
            "frame_id": self.frame_id,
        }

        ax, ay = self.compute_command(
            robot_state=self.robot_state,
            obstacles_states=self.obstacles_states,
            target_state=target_state,
        )

        self.publish_acceleration_command(ax, ay)

    # -------------------------------------------------
    # Obstacle reading
    # -------------------------------------------------

    def update_obstacles_states(self, now) -> List[dict]:
        obstacles_states = {}

        now_sec = now.nanoseconds * 1e-9

        for obstacle in self.obstacles.values():
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    obstacle["name"],
                    Time(),
                )

                x = tf_msg.transform.translation.x
                y = tf_msg.transform.translation.y


                vx, vy = self.estimate_obstacle_velocity(
                    name=obstacle["name"],
                    x=x,
                    y=y,
                    now_sec=now_sec
                )

                obstacles_states[obstacle["name"]] = {
                    "position": [x, y],
                    "velocity": [vx, vy],
                    "at": now_sec
                }

            except TransformException as ex:
                self.get_logger().warn(f"Could not read TF {self.frame_id} -> {obstacle['name']}: {ex}", throttle_duration_sec=1.0)

        self.obstacles_states = obstacles_states

    def estimate_obstacle_velocity(self, name, x, y, now_sec):
        if name not in self.obstacles_states.keys():
            return 0.0, 0.0

        px, py = self.obstacles_states[name]["position"]
        pt = self.obstacles_states[name]["at"]

        dt = now_sec - pt

        if dt <= 1e-6:
            return 0.0, 0.0

        vx = (x - px) / dt
        vy = (y - py) / dt

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

    def qp_objective(self, u, grad, u_ref):
        """
        Objective minimized by NLopt.
        """

        error = np.asarray(u, dtype=float) - u_ref

        if grad.size > 0:
            grad[:] = error

        return  0.5 * float(error @ error)

    def compute_command(self, robot_state, obstacles_states, target_state):
        # -------------------------------------------------
        # Current and target states
        # -------------------------------------------------
        p0 = np.asarray(robot_state["position"], dtype=float)
        v0 = np.asarray(robot_state["velocity"], dtype=float)

        pf = np.asarray(target_state["position"], dtype=float)
        vf = np.zeros(2, dtype=float)

        distance = np.linalg.norm(pf - p0)

        # if np.linalg.norm(v0) < 1e-3 and distance < self.target_tolerance:
        #     return 0.0, 0.0

        # -------------------------------------------------
        # Reference acceleration
        # -------------------------------------------------
        T = max(
            0.1,
            distance / self.reference_speed,
        )

        times = np.array([0.0, T], dtype=float)

        trajectory = CubicHermiteSpline(
            times,
            np.vstack((p0, pf)),
            np.vstack((v0, vf)),
            axis=0,
        )

        u_ref = np.asarray(
            trajectory(0.0, nu=2),
            dtype=float,
        )

        n_vars = 2

        acceleration_lower_bound = (
            -self.max_accel * np.ones(n_vars, dtype=float)
        )

        acceleration_upper_bound = (
            self.max_accel * np.ones(n_vars, dtype=float)
        )

        # -------------------------------------------------
        # QP objective
        #
        # min 0.5 * ||u - u_ref||²
        #
        # OSQP form:
        #
        # min 0.5 * u.T @ P @ u + q.T @ u
        #
        # therefore:
        #
        # P = I
        # q = -u_ref
        #
        # The constant 0.5 * u_ref.T @ u_ref is omitted.
        # -------------------------------------------------
        P = sparse.eye(
            n_vars,
            format="csc",
            dtype=float,
        )

        q = -u_ref

        # -------------------------------------------------
        # Double-integrator dynamics
        #
        # x_dot = F x + G u
        # -------------------------------------------------
        F = jnp.array(
            (
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            )
        )

        G = jnp.array(
            (
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            )
        )

        cbf_robot_state = jnp.asarray(
            np.concatenate((p0, v0)),
            dtype=float,
        )

        # -------------------------------------------------
        # OSQP constraints
        #
        # l <= A_qp @ u <= upper
        #
        # First two rows impose acceleration bounds:
        #
        # -max_accel <= u <= max_accel
        # -------------------------------------------------
        constraint_rows = [
            np.array([1.0, 0.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
        ]

        constraint_lower_bounds = [
            acceleration_lower_bound[0],
            acceleration_lower_bound[1],
        ]

        constraint_upper_bounds = [
            acceleration_upper_bound[0],
            acceleration_upper_bound[1],
        ]

        self.get_logger().info(
            f"Received obstacles_states: "
            f"type={type(obstacles_states).__name__}, "
            f"count={len(obstacles_states)}, "
            f"keys={list(obstacles_states.keys())}"
        )

        active_constraints = 0

        # -------------------------------------------------
        # CBF constraints
        # -------------------------------------------------
        for obstacle_name, obstacle_state in obstacles_states.items():
            self.get_logger().info(
                f"Preparing constraint for obstacle {obstacle_name}"
            )

            p_obs = np.asarray(
                obstacle_state["position"],
                dtype=float,
            )

            v_obs = np.asarray(
                obstacle_state["velocity"],
                dtype=float,
            )

            obstacle_distance = np.linalg.norm(p0 - p_obs)

            # This preserves your existing behavior.
            #
            # WARNING:
            # skipping the constraint when already close to an obstacle
            # may be unsafe. Consider removing this condition.
            if obstacle_distance <= 0.5:
                self.get_logger().warning(
                    f"Skipping {obstacle_name}: "
                    f"distance={obstacle_distance:.3f}"
                )
                continue

            cbf_obstacle_state = jnp.asarray(
                np.concatenate((p_obs, v_obs)),
                dtype=float,
            )

            n = 6
            tau = 1.2

            def h_as_function_of_robot_state(
                x,
                obstacle_state=cbf_obstacle_state,
                n_i=n,
                tau_i=tau,
            ):
                return compute_candidate_h(
                    x,
                    obstacle_state,
                    0.25,
                    0.25,
                    n_i,
                    tau_i,
                )

            h_value = h_as_function_of_robot_state(
                cbf_robot_state
            )

            grad_h = jax.grad(
                h_as_function_of_robot_state
            )(cbf_robot_state)

            class_k = class_K_function(
                h_value,
                gamma=100.0,
                beta=0,
            )

            # -------------------------------------------------
            # Original constraint:
            #
            # -(grad_h F x + grad_h G u + class_k) <= 0
            #
            # Equivalent CBF form:
            #
            # grad_h G u >= -(grad_h F x + class_k)
            #
            # OSQP representation:
            #
            # lower_i <= control_row @ u <= +inf
            # -------------------------------------------------
            control_row = np.asarray(
                grad_h @ G,
                dtype=float,
            ).reshape(n_vars)

            drift_and_class_k = float(
                grad_h @ F @ cbf_robot_state
                + class_k
            )

            cbf_lower_bound = -drift_and_class_k

            constraint_rows.append(control_row)
            constraint_lower_bounds.append(cbf_lower_bound)
            constraint_upper_bounds.append(np.inf)

            active_constraints += 1

            cbf_at_reference = (
                drift_and_class_k
                + control_row @ u_ref
            )

            self.get_logger().info(
                f"ob {obstacle_name} "
                f"(pos={p_obs}, vel={v_obs}) "
                f"h={float(h_value):.6f}, "
                f"CBF(u_ref)={cbf_at_reference:.6f} >= 0, "
                f"row={control_row}, "
                f"lower={cbf_lower_bound:.6f}"
            )

        # -------------------------------------------------
        # Assemble OSQP matrices
        # -------------------------------------------------
        A_qp = sparse.csc_matrix(
            np.vstack(constraint_rows),
            dtype=float,
        )

        lower = np.asarray(
            constraint_lower_bounds,
            dtype=float,
        )

        upper = np.asarray(
            constraint_upper_bounds,
            dtype=float,
        )

        # -------------------------------------------------
        # Setup and solve
        # -------------------------------------------------
        solver = osqp.OSQP()

        solver.setup(
            P=P,
            q=q,
            A=A_qp,
            l=lower,
            u=upper,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=self.nlopt_maxeval,
        )

        # Since a new solver is created every control iteration,
        # manually provide the previous solution as a warm start.
        previous_solution = getattr(
            self,
            "previous_solution",
            None,
        )

        if previous_solution is not None:
            previous_solution = np.asarray(
                previous_solution,
                dtype=float,
            )

            if previous_solution.shape == (n_vars,):
                solver.warm_start(
                    x=np.clip(
                        previous_solution,
                        acceleration_lower_bound,
                        acceleration_upper_bound,
                    )
                )

        results = solver.solve()

        status = results.info.status.lower()

        if (
            results.x is not None
            and status.startswith("solved")
        ):
            u_star = np.asarray(
                results.x,
                dtype=float,
            )

            self.previous_solution = u_star.copy()

        else:
            # This fallback is not guaranteed to satisfy the CBF constraints.
            u_star = np.clip(
                np.zeros(2),#u_ref,
                acceleration_lower_bound,
                acceleration_upper_bound,
            )

            self.get_logger().error(
                f"OSQP failed: "
                f"status={results.info.status}, "
                f"status_val={results.info.status_val}"
            )

        ax = float(u_star[0])
        ay = float(u_star[1])

        # Actual tracking cost, including the omitted constant.
        tracking_cost = 0.5 * np.dot(
            u_star - u_ref,
            u_star - u_ref,
        )

        self.get_logger().info(
            f"OSQP status={results.info.status}, "
            f"iterations={results.info.iter}, "
            f"primal_residual={results.info.prim_res:.3e}, "
            f"dual_residual={results.info.dual_res:.3e}, "
            f"active_constraints={active_constraints}, "
            f"cost={tracking_cost:.6f}, "
            f"u_ref=({u_ref[0]:.3f}, {u_ref[1]:.3f}), "
            f"cmd=({ax:.3f}, {ay:.3f})"
        )

        return ax, ay

    def compute_command_nlopt(self, robot_state, obstacles_states, target_state):
        p0 = np.asarray(robot_state["position"], dtype=float)
        v0 = np.asarray(robot_state["velocity"], dtype=float)

        pf = np.asarray(target_state["position"], dtype=float)
        vf = np.zeros(2)
        
        distance = np.linalg.norm(pf - p0)
        
        # if np.linalg.norm(v0) < 1e-3 and distance < self.target_tolerance:
        #     return 0.0, 0.0
        
        T = max(
            0.1,
            distance / self.reference_speed
        )

         # Time knots
        times = np.array([0.0, T])

        trajectory = CubicHermiteSpline(
            times,
            np.vstack((p0, pf)),
            np.vstack((v0, vf)),
            axis=0,
        )

        u_ref = np.asarray(trajectory(0.0, nu=2), dtype=float)
        
        # NLopt optimizer
        n_vars = 2
        opt = nlopt.opt(nlopt.LD_AUGLAG, n_vars)

        lower_bounds = -self.max_accel * np.ones(n_vars)
        upper_bounds = self.max_accel * np.ones(n_vars)

        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)

        opt.set_min_objective(
            lambda u, grad: self.qp_objective(
                u,
                grad,
                u_ref
            )
        )

        # Hard obstacle constraints
        # recall: NLopt inequality convention:
        #     g(u) <= 0
        A = jnp.array(((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
        G = jnp.array(((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

        # self.get_logger().info(f"{obstacles_states.values()}")
        self.get_logger().info(
            f"Received obstacles_states: "
            f"type={type(obstacles_states).__name__}, "
            f"count={len(obstacles_states)}, "
            f"keys={list(obstacles_states.keys())}"
        )


        for obstacle_name, obstacle_state in obstacles_states.items():
            self.get_logger().info(
                f"Preparing constraint for obstacle {obstacle_name}"
            )

            p_obs = np.asarray(obstacle_state["position"], dtype=float)
            v_obs = np.asarray(obstacle_state["velocity"], dtype=float)

            if np.linalg.norm(p0 - p_obs) <= 0.5:
                continue

            cbf_robot_state = jnp.asarray(
                np.concatenate((p0, v0)),
                dtype=float,
            )

            cbf_obstacle_state = jnp.asarray(
                np.concatenate((p_obs, v_obs)),
                dtype=float,
            )

            n = 6
            tau = 1.2

            def h_as_function_of_robot_state(
                x,
                obstacle_state=cbf_obstacle_state,
                n_i=n,
                tau_i=tau,
            ):
                return compute_candidate_h(
                    x,
                    obstacle_state,
                    0.25,
                    0.25,
                    n_i,
                    tau_i,
                )

            gradH = jax.grad(h_as_function_of_robot_state)

            class_k = class_K_function(
                h_as_function_of_robot_state(cbf_robot_state),
                gamma=100.0,
                beta=0,
            )

            grad_h = gradH(cbf_robot_state)

            def grad_const(
                u,
                grad_h_i=grad_h,
                robot_state_i=cbf_robot_state,
                class_k_i=class_k,
            ):
                return -(
                    grad_h_i @ A @ robot_state_i
                    + grad_h_i @ G @ u
                    + class_k_i
                )

            def gradgrad_const(
                u,
                grad_h_i=grad_h,
            ):
                return - grad_h_i @ G

            self.get_logger().info(
                f"ob {obstacle_name} "
                f"(pos: {p_obs}, vel: {v_obs}) "
                f"for u_ref({u_ref}): "
                f"{float(grad_const(jnp.asarray(u_ref))):.6f} <= 0 "
                f"({np.asarray(gradgrad_const(jnp.asarray(u_ref)))})"
            )

            def constraint(
                u,
                grad,
                value_function=grad_const,
                gradient_function=gradgrad_const,
                obstacle_name_i=obstacle_name,
            ):
                u_jax = jnp.asarray(u)

                value = value_function(u_jax)

                if grad.size > 0:
                    grad[:] = np.asarray(
                        gradient_function(u_jax),
                        dtype=float,
                    )

                return float(value)

            opt.add_inequality_constraint(
                constraint,
                1e-6,
            )   

            
        opt.set_maxeval(self.nlopt_maxeval)
        opt.set_xtol_rel(self.nlopt_xtol_rel)
        # opt.set_initial_step(0.25 * np.ones(n_vars))

        # -------------------------------------------------
        # Solve
        # -------------------------------------------------
        initial_guess = np.clip(u_ref, lower_bounds,  upper_bounds)
        u_star = np.zeros(2)
        
        try:
            u_star = opt.optimize(initial_guess)
        except Exception:
            pass
        
        result_code = opt.last_optimize_result()
        objective_value = opt.last_optimum_value()

        # self.previous_solution = np.array(u_star)

        ax = float(u_star[0])
        ay = float(u_star[1])

        # ax, ay = self.limit_acceleration(ax, ay)

        self.get_logger().info(
            f"NLopt result={result_code}, "
            f"cost={objective_value:.3f}, "
            f"cmd=({ax:.3f}, {ay:.3f})" # ,f"obstacles={len(obstacles)}",
            #, throttle_duration_sec=0.5
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