"""
Headless parameter sweeps of an agent crossing a sphere among moving obstacles.

The controlled agent crosses a sphere of radius SPHERE_RADIUS along a diameter,
from (-R, 0, 0) to (R, 0, 0), with the QP3DPrecomp controller. The obstacles
are not controlled: each one moves at constant velocity along a random chord of
the sphere, between two points at least 90 degrees apart, with a random speed,
and enters the sphere at a random time, so that obstacles cross the sphere
during the whole crossing of the agent. Contacts are disabled: the obstacles
keep their velocity and the safety of the agent is measured geometrically.

Studies:
    crossing  degree n and horizon tau of the super-hyperbola, over several
              layouts of the obstacles (seeds)
    speed     degree n against the speeds of the agent and of the obstacles,
              scaled together, on the same layouts at a fixed horizon

Each run saves its raw logs to data/sweep/<study>/<run_id>.mat. The analysis is
done in MATLAB (data/sweep/analyzeCrossing.m and data/sweep/analyzeSpeed.m).

Usage:
    python run_sweep.py [crossing|speed|all] [--workers W] [--overwrite]
"""

import argparse
import contextlib
import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# One thread per run, runs are parallelised by the pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import mujoco
import osqp
from scipy.io import savemat

from generate_scenarios import buildModel
from utils.utils import get_3d_position, get_3d_velocity
from controllers.qp_3d_precomp import QP3DPrecomp

SWEEP_DIR = Path(__file__).resolve().parent / "data" / "sweep"

# Agent and obstacles: point masses of 1 kg with three orthogonal forces
ROBOT_PATH = "scenarios/double_integrator/bot.xml"
COLLISION_RADIUS = 0.15
REF_SPEED = 0.2

# Sphere crossed by the agent along a diameter [m]
SPHERE_RADIUS = 2.0

# Range of the speeds of the obstacles [m/s], slower and faster than the agent
OBSTACLE_SPEEDS = (0.1, 0.4)

# Obstacles passing this close to the start of the agent [m] during the first
# START_CLEARANCE_TIME seconds are discarded: the agent cannot avoid them from rest
START_CLEARANCE = 0.6
START_CLEARANCE_TIME = 3.0

# Positions, velocities and accelerations are logged with this period [s]
LOG_PERIOD = 0.02

# The crossing is complete when the agent is this close to its target and slower than this
ARRIVAL_DISTANCE = 0.01
ARRIVAL_SPEED = 0.01

# Constraint i is active when L_f h + L_g h u + alpha(h) is below this (OSQP tolerance 1e-5)
ACTIVE_TOLERANCE = 1e-4

# With 60 obstacles about 29 are inside the sphere at a time, and an agent
# crossing in a straight line would hit 2.4 of them on average.
# a_max = 5 is the saturation of the actuators of bot.xml on a 1 kg body.
STUDIES = {
    "crossing": [
        {"M": 60, "n": n, "tau": tau, "a_max": 5.0, "gamma": 10.0, "dt": 0.01, "seed": seed}
        for n, tau, seed in itertools.product(
            (2, 4, 6, 8, 12),
            (0.5, 1.2, 2.5, 5.0),
            range(10),
        )
    ],
    # With "speed" = k the agent and the obstacles are k times faster on the
    # same layouts, at the horizon of the best crossings of the study above
    "speed": [
        {"M": 60, "n": n, "tau": 5.0, "a_max": 5.0, "gamma": 10.0, "dt": 0.01, "seed": seed, "speed": speed}
        for n, speed, seed in itertools.product(
            (2, 4, 6, 8, 12),
            (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
            range(10),
        )
    ],
}

# QP3DPrecomp caps OSQP at 100 iterations, the sweeps allow more so that they
# evaluate the CBF rather than the convergence of the solver
OSQP_MAX_ITER = 4000

_osqp_setup = osqp.OSQP.setup


def _setup_with_max_iter(self, *args, **kwargs):
    kwargs["max_iter"] = OSQP_MAX_ITER
    return _osqp_setup(self, *args, **kwargs)


osqp.OSQP.setup = _setup_with_max_iter

# Status and run time of the OSQP solutions of one compute_command: the CBF-QP
# and, when it is primal infeasible, the least-violation fallback
_osqp_infos = []
_osqp_solve = osqp.OSQP.solve


def _recording_solve(self, *args, **kwargs):
    results = _osqp_solve(self, *args, **kwargs)
    _osqp_infos.append((str(results.info.status).lower(), results.info.run_time))
    return results


osqp.OSQP.solve = _recording_solve


def run_id(params):
    speed = params.get("speed", 1.0)
    return (
        f"M{params['M']:03d}_n{params['n']:02d}_tau{params['tau']:.2f}"
        f"_gamma{params['gamma']:g}_dt{1000 * params['dt']:g}ms_seed{params['seed']}"
        + (f"_speed{speed:g}" if speed != 1.0 else "")
    )


def ref_speed(params):
    """
    Preferred speed of the agent. The optional parameter "speed" scales it and
    the speeds of the obstacles (1 by default, as in the studies).
    """
    return REF_SPEED * params.get("speed", 1.0)


def build_crossing(M, seed, speed=1.0):
    """
    Start and target of the agent and constant-velocity motion of M obstacles.

    The speeds of the agent and of the obstacles are scaled by speed, the times
    by 1 / speed: the positions and the chords of the obstacles, and so the
    layout of a seed, do not depend on speed.
    """
    rng = np.random.default_rng(seed)

    start = np.array([-SPHERE_RADIUS, 0.0, 0.0])
    target = -start
    crossing_time = 2.0 * SPHERE_RADIUS / (speed * REF_SPEED)

    entries, exits, speeds, enter_times, positions, velocities = [], [], [], [], [], []
    while len(positions) < M:
        entry = rng.normal(size=3)
        entry *= SPHERE_RADIUS / np.linalg.norm(entry)
        exit_ = rng.normal(size=3)
        exit_ *= SPHERE_RADIUS / np.linalg.norm(exit_)

        # Chords between points at least 90 degrees apart
        if entry @ exit_ > 0.0:
            continue

        obstacle_speed = speed * rng.uniform(*OBSTACLE_SPEEDS)
        chord = np.linalg.norm(exit_ - entry)
        velocity = obstacle_speed * (exit_ - entry) / chord

        # Entering the sphere from halfway along the chord at t = 0, up to the end of the crossing
        enter_time = rng.uniform(-0.5 * chord / obstacle_speed, crossing_time)
        position = entry - velocity * enter_time

        times = np.linspace(0.0, START_CLEARANCE_TIME / speed, 31)[:, None]
        if np.min(np.linalg.norm(position + times * velocity - start, axis=1)) < START_CLEARANCE:
            continue

        entries.append(entry)
        exits.append(exit_)
        speeds.append(obstacle_speed)
        enter_times.append(enter_time)
        positions.append(position)
        velocities.append(velocity)

    return start, target, {
        "entries": np.array(entries),
        "exits": np.array(exits),
        "speeds": np.array(speeds),
        "enter_times": np.array(enter_times),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
    }


def simulate(params, step_callback=None, prebuild_hook=None):
    """
    Simulates one crossing. step_callback, if given, is called at every step
    before the integration, with the model, the data and the step of the
    controller, and prebuild_hook, if given, edits the scene before it is
    compiled (both used by render_crossing.py).
    """
    M, n, tau, a_max = int(params["M"]), params["n"], params["tau"], params["a_max"]
    start, target, obstacles_motion = build_crossing(M, params["seed"], params.get("speed", 1.0))
    obstacle_names = [f"obstacle{j}" for j in range(M)]

    robots = [{
        "name": "agent",
        "collision_radius": COLLISION_RADIUS,
        "pos": tuple(start),
        "robot_path": ROBOT_PATH,
    }] + [{
        "name": name,
        "collision_radius": COLLISION_RADIUS,
        "pos": tuple(obstacles_motion["positions"][j]),
        "robot_path": ROBOT_PATH,
    } for j, name in enumerate(obstacle_names)]

    m, d, bindings, get_collision_spheres = buildModel(
        robots,
        [],
        prebuild_hook=prebuild_hook or (lambda builder: None),
        base_path="scenarios/double_integrator/base.xml",
        worldbody_path="scenarios/double_integrator/world.xml",
        assets_path="scenarios/double_integrator/assets.xml",
        defaults_path="scenarios/double_integrator/defaults.xml",
    )

    # No contacts: the obstacles cross each other keeping their velocity
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0

    # The controller runs at every step: the input is held over the step and
    # RK4 integrates the double integrator exactly
    m.opt.timestep = params["dt"]
    dt = m.opt.timestep

    # The obstacles are never actuated and keep their initial velocity
    for j, name in enumerate(obstacle_names):
        dof = m.body_dofadr[bindings[name]["bodies"]["robot"]]
        d.qvel[dof:dof + 3] = obstacles_motion["velocities"][j]

    mujoco.mj_forward(m, d)

    agent_body = bindings["agent"]["bodies"]["robot"]
    actuators = [bindings["agent"]["actuators"][f"force_{axis}"] for axis in "xyz"]

    log_every = max(1, round(LOG_PERIOD / dt))
    t_max = 30.0 + 4.0 * 2.0 * SPHERE_RADIUS / ref_speed(params)
    max_steps = int(np.ceil(t_max / dt))

    def read_state():
        return np.r_[get_3d_position(d, agent_body), get_3d_velocity(d, agent_body)]

    state = read_state()

    controller = QP3DPrecomp(
        dt=dt,
        target=target,
        initial_state=state,
        sh_n=n,
        sh_tau=tau,
        cbf_gamma=params["gamma"],
        collision_radius=COLLISION_RADIUS,
        obstacles=get_collision_spheres(["agent"]),
    )
    controller.set_max_accel(a_max)
    controller.set_reference_speed(ref_speed(params))

    # Per step, for the agent (one column, as for one agent among N)
    shape = (max_steps, 1)
    gap_nn = np.full(shape, np.nan, np.float32)
    h_min = np.full(shape, np.nan, np.float32)
    n_active = np.zeros(shape, np.int16)
    qp_solved = np.zeros(shape, np.uint8)
    qp_infeasible = np.zeros(shape, np.uint8)
    t_command = np.zeros(shape, np.float32)
    t_osqp = np.zeros(shape, np.float32)
    du_norm = np.zeros(shape, np.float32)

    # Every log_every steps
    log_rows = max_steps // log_every + 1
    t_log = np.zeros(log_rows)
    p_log = np.zeros((log_rows, 1, 3), np.float32)
    v_log = np.zeros((log_rows, 1, 3), np.float32)
    u_log = np.zeros((log_rows, 1, 3), np.float32)
    u_ref_log = np.zeros((log_rows, 1, 3), np.float32)
    obstacle_p_log = np.zeros((log_rows, M, 3), np.float32)

    wall_start = time.perf_counter()
    arrived = False
    k = -1

    for k in range(max_steps):
        state = read_state()

        obstacles = get_collision_spheres(["agent"])
        obstacle_positions = np.array([obstacles[name]["p"] for name in obstacle_names])
        gap_nn[k, 0] = np.min(np.linalg.norm(obstacle_positions - state[:3], axis=1)) - 2.0 * COLLISION_RADIUS

        controller.update_state(state)
        controller.update_obstacles(obstacles)
        reference = controller.compute_acceleration_reference()

        _osqp_infos.clear()
        start_ns = time.perf_counter()
        command, boundaries = controller.compute_command()
        t_command[k, 0] = time.perf_counter() - start_ns

        command = np.asarray(command, dtype=float)
        status = _osqp_infos[0][0]
        qp_solved[k, 0] = status.startswith("solved")
        qp_infeasible[k, 0] = status.startswith("primal infeasible")
        t_osqp[k, 0] = sum(run_time for _, run_time in _osqp_infos)
        controller.increment_step()

        grads = np.array([boundary["grad_h_value"] for boundary in boundaries.values()])
        h_values = np.array([boundary["h_value"] for boundary in boundaries.values()])
        velocities = np.array([obstacles[name]["v"] for name in boundaries])
        conditions = (
            np.einsum("ij,ij->i", grads[:, :3], state[3:] - velocities)
            + grads[:, 3:] @ command
            + params["gamma"] * h_values
        )
        n_active[k, 0] = np.count_nonzero(conditions < ACTIVE_TOLERANCE)
        h_min[k, 0] = h_values.min()

        for axis in range(3):
            d.ctrl[actuators[axis]] = command[axis]

        du_norm[k, 0] = np.linalg.norm(command - reference)

        if step_callback is not None:
            step_callback(m, d, {
                "bindings": bindings,
                "state": state,
                "target": target,
                "command": command,
                "reference": reference,
                "obstacle_names": obstacle_names,
                "obstacle_positions": obstacle_positions,
                "active": dict(zip(boundaries, conditions < ACTIVE_TOLERANCE)),
                "gap": float(gap_nn[k, 0]),
            })

        if k % log_every == 0:
            row = k // log_every
            t_log[row] = d.time
            p_log[row, 0], v_log[row, 0] = state[:3], state[3:]
            u_log[row, 0], u_ref_log[row, 0] = command, reference
            obstacle_p_log[row] = obstacle_positions

        arrived = (
            np.linalg.norm(state[:3] - target) < ARRIVAL_DISTANCE
            and np.linalg.norm(state[3:]) < ARRIVAL_SPEED
        )

        mujoco.mj_step(m, d)

        if arrived:
            break

    steps = k + 1
    rows = (steps - 1) // log_every + 1

    return {
        "params": {key: float(value) for key, value in params.items()},
        "run_id": run_id(params),
        "dt": dt,
        "t_max": t_max,
        "completed": bool(arrived),
        "sphere_radius": SPHERE_RADIUS,
        "collision_radius": COLLISION_RADIUS,
        "radius_tolerance": controller.radius_tollerance,
        "ref_speed": ref_speed(params),
        "slowing_distance": controller.slowing_distance,
        "velocity_time_constant": controller.velocity_time_constant,
        "osqp_max_iter": OSQP_MAX_ITER,
        "starts": start[None, :],
        "targets": target[None, :],
        "obstacle_starts": obstacles_motion["positions"],
        "obstacle_velocities": obstacles_motion["velocities"],
        "obstacle_entries": obstacles_motion["entries"],
        "obstacle_exits": obstacles_motion["exits"],
        "obstacle_speeds": obstacles_motion["speeds"],
        "obstacle_enter_times": obstacles_motion["enter_times"],
        "time": np.arange(steps) * dt,
        "gap_nn": gap_nn[:steps],
        "h_min": h_min[:steps],
        "n_active": n_active[:steps],
        "qp_solved": qp_solved[:steps],
        "qp_infeasible": qp_infeasible[:steps],
        "t_command": t_command[:steps],
        "t_osqp": t_osqp[:steps],
        "du_norm": du_norm[:steps],
        "log_every": log_every,
        "t_log": t_log[:rows],
        "p": p_log[:rows],
        "v": v_log[:rows],
        "u": u_log[:rows],
        "u_ref": u_ref_log[:rows],
        "obstacle_p": obstacle_p_log[:rows],
        "wall_time": time.perf_counter() - wall_start,
    }


def run_and_save(job):
    params, path = job

    # The controller prints its OSQP status, keep the terminal for the progress
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        result = simulate(params)

    savemat(path, result, do_compression=True)

    return (
        f"{result['run_id']}: {'completed' if result['completed'] else 'NOT completed'} "
        f"at t={result['time'][-1]:.2f}s, min gap {np.nanmin(result['gap_nn']):+.4f} m, "
        f"{result['wall_time']:.0f}s wall"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", nargs="?", default="all", choices=[*STUDIES, "all"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2 - 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = []
    for study in (STUDIES if args.study == "all" else [args.study]):
        study_dir = SWEEP_DIR / study
        study_dir.mkdir(parents=True, exist_ok=True)

        with open(study_dir / "manifest.csv", "w", newline="") as manifest:
            writer = csv.writer(manifest)
            writer.writerow(["run_id", *STUDIES[study][0].keys()])
            for params in STUDIES[study]:
                writer.writerow([run_id(params), *params.values()])

        for params in STUDIES[study]:
            path = study_dir / f"{run_id(params)}.mat"
            if args.overwrite or not path.exists():
                jobs.append((params, path))

    print(f"{len(jobs)} runs with {args.workers} workers", file=sys.stderr)

    with Pool(args.workers) as pool:
        for index, message in enumerate(pool.imap_unordered(run_and_save, jobs), start=1):
            print(f"[{index}/{len(jobs)}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
