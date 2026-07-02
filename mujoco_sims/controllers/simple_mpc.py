import threading
import time

import do_mpc
import numpy as np
from casadi import *

from utils.utils import wrap


class SimpleMPC:
    """
    Velocity-controlled MPC.

    State:
        x = [x, y, yaw]

    Input:
        u = [v, w]
    """

    def __init__(
        self,
        target=np.array([1.0, 1.0]),
        initial_state=np.zeros(3),
        obstacles=[]
    ):

        self.target = target
        self.obs_nl_cons = {}

        self.state_lock = threading.Lock()

        self.state = np.asarray(initial_state, dtype=float)

        # commanded [v, w]
        self.velocity_command = np.zeros(2)

        # ==========================================================
        # MODEL
        # ==========================================================

        model = do_mpc.model.Model("continuous")

        x = model.set_variable("_x", "x")
        y = model.set_variable("_x", "y")
        yaw = model.set_variable("_x", "yaw")

        v = model.set_variable("_u", "v")
        w = model.set_variable("_u", "w")

        target_x = model.set_variable("_p", "target_x")
        target_y = model.set_variable("_p", "target_y")


        for obstacle_name, _ in obstacles.items():

            obs_x = "obs_" + obstacle_name + "_x"
            obs_y = "obs_" + obstacle_name + "_y"
            obs_rad = "obs_" + obstacle_name + "_rad"

            obstacle_x = model.set_variable("_p", obs_x)
            obstacle_y = model.set_variable("_p", obs_y)
            obstacle_rad = model.set_variable("_p", obs_rad)

        self.obstacles = obstacles

        model.set_rhs("x", v * cos(yaw))
        model.set_rhs("y", v * sin(yaw))
        model.set_rhs("yaw", w)

        model.setup()

        self.model = model

        # ==========================================================
        # MPC
        # ==========================================================

        self.mpc = do_mpc.controller.MPC(model)

        self.mpc.set_param(
            n_horizon=10,
            t_step=0.1,
            state_discretization="collocation",
            store_full_solution=True,
        )

        target_heading = atan2(
            model.p["target_y"] - model.x["y"],
            model.p["target_x"] - model.x["x"]
        )

        heading_error = atan2(
            sin(target_heading - model.x["yaw"]),
            cos(target_heading - model.x["yaw"])
        )

        # terminal cost - distance to target
        mterm = (
            (model.x["x"] - model.p["target_x"]) ** 2
            + (model.x["y"] - model.p["target_y"]) ** 2
        )

        # running cost
        lterm = (
            mterm
            + 0.01 * model.u["v"] ** 2
            + 0.01 * model.u["w"] ** 2
            + 0.01 * heading_error ** 2
        )

        self.mpc.set_objective(
            mterm=mterm,
            lterm=lterm
        )

        # penalize command changes
        self.mpc.set_rterm(
            v=5.00,
            w=0.05
        )

        for obstacle_name, _ in self.obstacles.items():
            obs_x = "obs_" + obstacle_name + "_x"
            obs_y = "obs_" + obstacle_name + "_y"
            obs_rad = "obs_" + obstacle_name + "_rad"

            obs_nl_cons = (
                (x - model.p[obs_x])**2
                + (y - model.p[obs_y])**2
                - (0.14 + model.p[obs_rad] + 0.01)**2
            )

            self.mpc.set_nl_cons(
                "constraint_"+obstacle_name,
                -obs_nl_cons,
                ub=0
            )

        self.mpc.settings.supress_ipopt_output()

        # Params setup
        p_template = self.mpc.get_p_template(1)

        def p_fun(t_now):
            p_template["_p", 0, "target_x"] = self.target[0]
            p_template["_p", 0, "target_y"] = self.target[1]


            for obstacle_name, obstacle in self.obstacles.items():
                obs_x = "obs_" + obstacle_name +"_x"
                obs_y = "obs_" + obstacle_name +"_y"
                obs_rad = "obs_" + obstacle_name +"_rad"

                p_template["_p", 0, obs_x] = obstacle["p"][0]
                p_template["_p", 0, obs_y] = obstacle["p"][1]
                p_template["_p", 0, obs_rad] = obstacle["collision_radius"]
                
            # print(p_template["_p", 0])
            return p_template

        self.mpc.set_p_fun(p_fun)

        # ==========================================================
        # BOUNDS
        # ==========================================================

        self.mpc.bounds["lower", "_u", "v"] = 0.0
        self.mpc.bounds["upper", "_u", "v"] = 0.1

        self.mpc.bounds["lower", "_u", "w"] = -1.0
        self.mpc.bounds["upper", "_u", "w"] = 1.0

        self.mpc.setup()

        # ==========================================================
        # INITIALIZATION
        # ==========================================================

        self.mpc.x0 = self.state
        self.mpc.set_initial_guess()

        # ==========================================================
        # THREAD
        # ==========================================================

        self.running = True

        self.thread = threading.Thread(
            target=self.mpc_thread_fn,
            daemon=True
        )

        self.thread.start()

    # ==============================================================
    # STATE UPDATE
    # ==============================================================

    def update_state(self, x, y, yaw):
        with self.state_lock:
            self.state = np.array([x, y, yaw])

    def set_target(self, target):
        self.target = target

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles

    # ==============================================================
    # GET COMMAND
    # ==============================================================

    def get_command(self):
        return self.velocity_command.copy()

    # ==============================================================
    # MPC LOOP
    # ==============================================================

    def mpc_thread_fn(self):

        print("[MPC] thread started")

        period = 1/10  # 15 Hz

        while self.running:

            start = time.time()

            with self.state_lock:
                x0 = self.state.copy()
                print(f"[MPC] state: {x0}")
                print(f"[MPC] target: {self.target}")

            try:
                start = time.time()
                u = self.mpc.make_step(x0)

                elapsed = time.time()-start

                self.velocity_command[0] = float(u[0])
                self.velocity_command[1] = float(u[1])

                print(f"[MPC][DEBUG] took: {elapsed:.4f}s")
                # print(f"[MPC][DEBUG] success: {self.mpc.data['success']}, t_wall_total: {self.mpc.data['t_wall_total']}")

            except Exception as e:

                print(f"[MPC] error: {e}")

            elapsed = time.time() - start

            time.sleep(max(0.0, period - elapsed))

    # ==============================================================
    # STOP
    # ==============================================================

    def stop(self):
        self.running = False
