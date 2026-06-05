import threading
import time

import do_mpc
import numpy as np
from casadi import *


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
        initial_state=np.zeros(3)
    ):

        self.target = np.asarray(target, dtype=float)

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
            n_horizon=20,
            t_step=0.05,
            state_discretization="collocation",
            store_full_solution=True,
        )

        # terminal cost - distance to target
        mterm = (
            (model.x["x"] - self.target[0]) ** 2
            + (model.x["y"] - self.target[1]) ** 2
        )

        # running cost
        lterm = (
            mterm
            + 0.01 * model.u["v"] ** 2
            + 0.01 * model.u["w"] ** 2
        )

        self.mpc.set_objective(
            mterm=mterm,
            lterm=lterm
        )


        # penalize command changes
        self.mpc.set_rterm(
            v=0.05,
            w=0.05
        )

        # ==========================================================
        # BOUNDS
        # ==========================================================

        self.mpc.bounds["lower", "_u", "v"] = 0.0
        self.mpc.bounds["upper", "_u", "v"] = 0.1

        self.mpc.bounds["lower", "_u", "w"] = -3.0
        self.mpc.bounds["upper", "_u", "w"] = 3.0

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

        period = 0.05  # 20 Hz

        while self.running:

            start = time.time()

            with self.state_lock:
                x0 = self.state.copy()

            try:

                u = self.mpc.make_step(x0)

                self.velocity_command[0] = float(u[0])
                self.velocity_command[1] = float(u[1])

            except Exception as e:

                print(f"[MPC] error: {e}")

            elapsed = time.time() - start

            time.sleep(max(0.0, period - elapsed))

    # ==============================================================
    # STOP
    # ==============================================================

    def stop(self):
        self.running = False