"""
Videos of crossings of run_sweep.py, rendered offscreen with MuJoCo.

Each case (degree n and horizon tau) is simulated again on the same layout of
the obstacles and rendered to data/crossing_video/:

    - the agent in blue, with its path, its velocity (green) and its commanded
      acceleration (yellow), both scaled by ARROW_SCALE;
    - the obstacles with their collision radius, only while inside the sphere,
      in orange when a CBF constraint of the agent on them is active;
    - the time, the case and the gap to the closest obstacle;
    - on the right, the velocity space of the obstacle in red, the one with the
      lowest CBF: the collision set of the case, the boundary of another degree
      for comparison, the FVO and the relative velocity of the agent.

With --side-by-side and several cases, the videos are also stacked side by side.
With --snapshots, the 3D view at the given times is also saved as a PNG, without
the texts and cropped to the sphere, with the velocity space of the panel in a
.mat file (for data/sweep/paperFigures.m).
With --speed k, the agent and the obstacles are k times faster than in the
studies of run_sweep.py, on the same layout of the obstacles. The background is
white, or black with --background black.

Usage:
    python render_crossing.py [--seed 4] [--n 4] [--tau 0.5 5] [--speed 1] [--side-by-side]
    python render_crossing.py --seed 8 --n 2 4 --tau 5 --speed 2.5 --side-by-side
    python render_crossing.py --seed 3 --n 2 4 --tau 5 --speed 2.5 --snapshots 5
"""

import argparse
import contextlib
import inspect
import io
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.io import savemat

import run_sweep
from controllers.qp_3d_precomp import QP3DPrecomp
from controllers.sh_cbf_core import compute_and_eval_h_and_grad_batch, compute_sh_parameters
from utils.utils import Rz, draw_sphere

VIDEO_DIR = Path(__file__).resolve().parent / "data" / "crossing_video"

FPS = 30
WIDTH = 960
HEIGHT = 720

# Velocity and acceleration arrows are drawn with this length per m/s and m/s^2 [s, s^2]
ARROW_SCALE = 3.0
ARROW_WIDTH = 0.012

CAMERA = {"lookat": (0.0, 0.0, 0.0), "distance": 7.0, "azimuth": 120.0, "elevation": -22.0}

# Rows and columns of the 3D view in the snapshots: the sphere, centred in the view
SNAPSHOT_CROP = (slice(80, 640), slice(200, 760))

# Radius of the agent in its CBFs, with the margin of QP3DPrecomp
AGENT_RADIUS = run_sweep.COLLISION_RADIUS + inspect.signature(QP3DPrecomp).parameters["radius_tollerance"].default

AGENT_COLOR = (0.00, 0.45, 0.74, 1.0)
ACTIVE_COLOR = (0.95, 0.50, 0.10, 0.85)
SELECTED_COLOR = (0.90, 0.10, 0.10, 0.90)
HIDDEN = (0.0, 0.0, 0.0, 0.0)

# Colours that depend on the background: RGBA in the 3D view, BGR or grey
# levels in the texts and in the panel. Without a skybox MuJoCo clears to
# black, so the white background is a flat skybox added to the scene. The
# obstacles are semi-transparent, so that the agent stays visible behind them.
BACKGROUNDS = {
    "white": {
        "skybox": "1 1 1", "haze": (1.0, 1.0, 1.0, 1.0),
        "sphere": (0.30, 0.45, 0.75, 0.08), "obstacle": (0.55, 0.55, 0.55, 0.55),
        "acceleration": (0.90, 0.60, 0.00, 0.95), "overlay": (30, 30, 30),
        "panel": 255, "text": (30, 30, 30), "secondary": (90, 90, 90), "ticks": (110, 110, 110),
        "grid": (225, 225, 225), "axis": (150, 150, 150), "frame": (170, 170, 170),
        "fvo": (30, 30, 30), "preferred": (30, 30, 30),
    },
    "black": {
        "skybox": None, "haze": None,
        "sphere": (0.55, 0.70, 0.95, 0.06), "obstacle": (0.62, 0.62, 0.62, 0.45),
        "acceleration": (1.0, 0.85, 0.0, 0.9), "overlay": (255, 255, 255),
        "panel": 20, "text": (230, 230, 230), "secondary": (170, 170, 170), "ticks": (130, 130, 130),
        "grid": (45, 45, 45), "axis": (110, 110, 110), "frame": (90, 90, 90),
        "fvo": (235, 235, 235), "preferred": (255, 255, 255),
    },
}

# Panel of the velocity space, on the right of the 3D view
PANEL_WIDTH = 400
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Plotted relative velocities [m/s], across (v_t) and along (v_r) the line of
# sight, for the speeds of the studies (scaled by --speed)
PLOT_VT = 0.5
PLOT_VR = (-0.15, 0.65)
PLOT_SCALE = 360  # px per m/s
PLOT_ORIGIN = (20, 90)  # top left corner in the panel [px]

# BGR colours of the collision sets, per degree
SET_COLORS = {2: (220, 100, 220), 4: (220, 200, 40), 6: (90, 200, 90), 8: (60, 150, 255), 12: (200, 200, 200)}
RELATIVE_VELOCITY_COLOR = (25, 204, 25)

# The obstacle in the panel changes only when another one has a CBF lower by this [m/s]
SELECTION_HYSTERESIS = 0.02

# Preferred velocity of the agent, as in QP3DPrecomp
SLOWING_DISTANCE = 0.2


class CrossingRenderer:
    """Step callback of run_sweep.simulate that writes one video frame every 1/FPS s."""

    def __init__(self, path, label, params, other_n, snapshots=(), background="white"):
        self.path = Path(path)
        self.theme = BACKGROUNDS[background]
        self.snapshots = sorted(snapshots)
        self.label = label
        self.n = params["n"]
        self.tau = params["tau"]
        self.speed = params.get("speed", 1.0)
        self.ref_speed = run_sweep.ref_speed(params)
        self.other_n = other_n
        self.selected = None
        self.renderer = None
        self.writer = None
        self.next_frame = 0.0
        self.trail = []
        self.frames = 0

    def setup(self, m, d, info):
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, WIDTH)
        m.vis.global_.offheight = max(m.vis.global_.offheight, HEIGHT)
        self.renderer = mujoco.Renderer(m, width=WIDTH, height=HEIGHT, max_geom=10_000)

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = CAMERA["lookat"]
        self.camera.distance = CAMERA["distance"]
        self.camera.azimuth = CAMERA["azimuth"]
        self.camera.elevation = CAMERA["elevation"]

        bindings = info["bindings"]
        self.agent_geom = bindings["agent"]["geom"]["robot_geom"]
        self.obstacle_geoms = [bindings[name]["geom"]["robot_geom"] for name in info["obstacle_names"]]

        # The obstacles keep their initial velocity
        dofs = [m.body_dofadr[bindings[name]["bodies"]["robot"]] for name in info["obstacle_names"]]
        self.obstacle_velocities = np.array([d.qvel[dof:dof + 3] for dof in dofs])

        # Every robot drawn with its collision radius, no floor
        for geom in [self.agent_geom, *self.obstacle_geoms]:
            m.geom_size[geom, 0] = run_sweep.COLLISION_RADIUS
        m.geom_rgba[self.agent_geom] = AGENT_COLOR
        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")] = HIDDEN
        if self.theme["haze"] is not None:
            m.vis.rgba.haze[:] = self.theme["haze"]

        # The centre marker of bot.xml would show the hidden obstacles as white dots
        m.site_rgba[:] = HIDDEN

        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH + PANEL_WIDTH, HEIGHT)
        )

    def __call__(self, m, d, info):
        if self.renderer is None:
            self.setup(m, d, info)

        if d.time + 1e-9 < self.next_frame:
            return
        self.next_frame += 1.0 / FPS

        state = info["state"]
        self.trail.append(state[:3].copy())

        inside = np.linalg.norm(info["obstacle_positions"], axis=1) <= run_sweep.SPHERE_RADIUS
        self.select_obstacle(state, info["obstacle_positions"], inside)

        # Obstacles inside the sphere, in red the one of the panel, in orange when
        # a constraint on them is active
        for j, (name, geom) in enumerate(zip(info["obstacle_names"], self.obstacle_geoms)):
            if not inside[j]:
                m.geom_rgba[geom] = HIDDEN
            elif j == self.selected:
                m.geom_rgba[geom] = SELECTED_COLOR
            elif info["active"].get(name, False):
                m.geom_rgba[geom] = ACTIVE_COLOR
            else:
                m.geom_rgba[geom] = self.theme["obstacle"]

        self.renderer.update_scene(d, camera=self.camera)
        scene = self.renderer.scene

        draw_sphere(scene, np.zeros(3), self.theme["sphere"], run_sweep.SPHERE_RADIUS)
        draw_sphere(scene, info["target"], (0.20, 0.75, 0.25, 1.0), 0.06)
        for point in self.trail[::2]:
            draw_sphere(scene, point, AGENT_COLOR, 0.015)

        for vector, color in ((state[3:], (0.1, 0.8, 0.1, 0.9)), (info["command"], self.theme["acceleration"])):
            norm = np.linalg.norm(vector)
            if norm > 1e-4:
                start = state[:3] + vector / norm * run_sweep.COLLISION_RADIUS
                draw_arrow(scene, start, ARROW_SCALE * vector, color)

        frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        space = self.velocity_space(state, info)
        if self.snapshots and d.time + 1e-9 >= self.snapshots[0]:
            self.snapshots.pop(0)
            self.save_snapshot(frame, space, d.time)

        for row, text in enumerate((
            self.label,
            f"t = {d.time:5.1f} s",
            f"gap to the closest obstacle = {100 * info['gap']:5.1f} cm",
        )):
            cv2.putText(frame, text, (20, 40 + 32 * row), FONT, 0.8, self.theme["overlay"], 2, cv2.LINE_AA)

        self.writer.write(np.hstack((frame, self.draw_panel(space))))
        self.frames += 1

    def select_obstacle(self, state, positions, inside):
        """
        Selects the obstacle of the panel, the one inside the sphere with the
        lowest CBF, and keeps the CBFs of all the obstacles in self.h.
        """
        _, _, _, h, _ = compute_and_eval_h_and_grad_batch(
            state,
            np.c_[positions, self.obstacle_velocities],
            AGENT_RADIUS,
            run_sweep.COLLISION_RADIUS,
            self.n,
            self.tau,
        )
        self.h = h
        lowest = int(np.argmin(np.where(inside, h, np.inf)))

        if not inside[lowest]:
            self.selected = None
        elif (self.selected is None or not inside[self.selected]
              or h[lowest] < h[self.selected] - self.speed * SELECTION_HYSTERESIS):
            self.selected = lowest

    def velocity_space(self, state, info):
        """
        Velocity space of the selected obstacle B, in the frame of the line of
        sight: v_r along e = (p_B - p_A) / ell and v_t across it. The collision
        sets are rotationally symmetric about e, so a relative velocity is in a
        set if it is above its boundary.

        Returns None without a selected obstacle, otherwise the distance ell and
        the radius R of the pair, the CBF h of the selected obstacle, (v_t, v_r)
        of the relative velocity and of the preferred velocity relative to B
        and, if apart, the parameters a and b of this case and of other_n.
        """
        if self.selected is None:
            return None

        j = self.selected
        R = AGENT_RADIUS + run_sweep.COLLISION_RADIUS
        delta_p = info["obstacle_positions"][j] - state[:3]
        ell = np.linalg.norm(delta_p)
        e_los = delta_p / ell

        error = info["target"] - state[:3]
        preferred = self.ref_speed * error / max(np.linalg.norm(error), SLOWING_DISTANCE)
        velocities = [state[3:] - self.obstacle_velocities[j], preferred - self.obstacle_velocities[j]]
        relative, preferred = (np.r_[np.linalg.norm(v - (v @ e_los) * e_los), v @ e_los] for v in velocities)

        space = {"ell": ell, "R": R, "h": self.h[j], "relative": relative, "preferred": preferred}
        if ell > R:
            space["a"], space["b"], _, _ = compute_sh_parameters(ell, R, self.n, self.tau)
            space["a_other"], space["b_other"], _, _ = compute_sh_parameters(ell, R, self.other_n, self.tau)
        return space

    def save_snapshot(self, frame, space, time):
        """3D view without the texts, cropped to the sphere, and its velocity space."""
        stem = str(self.path.with_name(f"{self.path.stem}_t{time:05.2f}s"))
        cv2.imwrite(stem + ".png", frame[SNAPSHOT_CROP])
        savemat(stem + ".mat", {
            "time": time, "n": self.n, "other_n": self.other_n, "tau": self.tau, "speed": self.speed,
            "selected": -1 if space is None else self.selected, **(space or {}),
        })
        print(f"saved {stem}.png and .mat")

    def draw_panel(self, space):
        """The velocity space of the selected obstacle, as returned by velocity_space."""
        theme = self.theme
        panel = np.full((HEIGHT, PANEL_WIDTH, 3), theme["panel"], np.uint8)
        put_text(panel, "Velocity space of the red obstacle", (20, 36), 0.6, theme["text"])
        put_text(panel, "along (v_r) and across (v_t) the line of sight", (20, 62), 0.45, theme["secondary"])

        plot_width = round(2 * PLOT_VT * PLOT_SCALE)
        plot_height = round((PLOT_VR[1] - PLOT_VR[0]) * PLOT_SCALE)
        plot = np.full((plot_height, plot_width, 3), theme["panel"], np.uint8)

        # Grid in the velocities of the studies, labelled with the scaled ones
        s = self.speed
        for v_t in np.arange(-0.4, 0.41, 0.2):
            color = theme["axis"] if abs(v_t) < 1e-9 else theme["grid"]
            cv2.polylines(plot, [to_px([v_t, v_t], PLOT_VR)], False, color, 1, cv2.LINE_AA, shift=4)
        for v_r in (0.0, 0.2, 0.4, 0.6):
            color = theme["axis"] if v_r == 0.0 else theme["grid"]
            cv2.polylines(plot, [to_px([-PLOT_VT, PLOT_VT], [v_r, v_r])], False, color, 1, cv2.LINE_AA, shift=4)
            put_text(plot, f"{s * v_r:.1f}", (4, int(to_px(0.0, v_r)[0, 1] / 16) - 4), 0.4, theme["ticks"])
        put_text(plot, "v_r [m/s]", (plot_width // 2 + 6, 16), 0.45, theme["secondary"])

        readouts = []
        if space is not None:
            ell, R = space["ell"], space["R"]
            if ell > R:
                b_own, b_other = space["b"], space["b_other"]
                own = boundary(space["a"], b_own, self.n, s)
                other = boundary(space["a_other"], b_other, self.other_n, s)

                # Collision set of this case, filled
                fill = plot.copy()
                top = s * (PLOT_VR[1] + 0.1)
                cv2.fillPoly(fill, [to_px(np.r_[own[0], s * PLOT_VT, -s * PLOT_VT], np.r_[own[1], top, top], s)],
                             SET_COLORS.get(self.n, (200, 200, 200)), cv2.LINE_AA, shift=4)
                cv2.addWeighted(fill, 0.35, plot, 0.65, 0.0, dst=plot)

                # FVO: the cut-off disk D(p / tau, R / tau) and the cone tangent to it
                cv2.circle(plot, tuple(to_px(0.0, ell / self.tau, s)[0]), round(16 * PLOT_SCALE * R / (s * self.tau)),
                           theme["fvo"], 1, cv2.LINE_AA, shift=4)
                sin_theta = R / ell
                cos_theta = np.sqrt(1.0 - sin_theta**2)
                tangency = np.sqrt(ell**2 - R**2) / self.tau
                for side in (-1.0, 1.0):
                    far = np.r_[tangency, 2.0 * s]
                    cv2.polylines(plot, [to_px(side * sin_theta * far, cos_theta * far, s)],
                                  False, theme["fvo"], 1, cv2.LINE_AA, shift=4)

                cv2.polylines(plot, [to_px(*other, s)], False, SET_COLORS.get(self.other_n, (200, 200, 200)), 2,
                              cv2.LINE_AA, shift=4)
                cv2.polylines(plot, [to_px(*own, s)], False, SET_COLORS.get(self.n, (200, 200, 200)), 3,
                              cv2.LINE_AA, shift=4)

                readouts = [
                    f"distance = {ell:.2f} m,  h = {space['h']:+.3f} m/s",
                    f"b = {b_own:.3f} (n = {self.n}),  {b_other:.3f} (n = {self.other_n}) m/s",
                ]
            else:
                readouts = [f"distance = {ell:.2f} m: overlapping"]

            relative, preferred = (tuple(to_px(*space[key], s)[0]) for key in ("relative", "preferred"))
            cv2.circle(plot, preferred, 16 * 6, theme["preferred"], 2, cv2.LINE_AA, shift=4)
            cv2.circle(plot, relative, 16 * 6, RELATIVE_VELOCITY_COLOR, -1, cv2.LINE_AA, shift=4)
        else:
            put_text(plot, "no obstacle in the sphere", (20, plot_height // 2), 0.5, theme["text"])

        x0, y0 = PLOT_ORIGIN
        panel[y0:y0 + plot_height, x0:x0 + plot_width] = plot
        cv2.rectangle(panel, (x0, y0), (x0 + plot_width - 1, y0 + plot_height - 1), theme["frame"], 1)
        for v_t in np.arange(-0.4, 0.41, 0.2):
            x = x0 + int(to_px(v_t, 0.0)[0, 0] / 16)
            put_text(panel, f"{s * v_t:+.1f}" if abs(v_t) > 1e-9 else "0", (x - 14, y0 + plot_height + 18), 0.4,
                     theme["ticks"])
        put_text(panel, "v_t [m/s]", (x0 + plot_width // 2 - 36, y0 + plot_height + 42), 0.45, theme["secondary"])

        # Legend
        y = y0 + plot_height + 80
        rows = [
            ("fill", self.n, f"n = {self.n}: {set_name(self.n)} (this case)"),
            ("line", self.other_n, f"n = {self.other_n}: {set_name(self.other_n)}"),
            ("fvo", None, "FVO: cut-off disk and cone"),
            ("relative", None, "relative velocity v_A - v_B"),
            ("preferred", None, "preferred velocity - v_B"),
        ]
        for kind, n, text in rows:
            color = SET_COLORS.get(n, (200, 200, 200))
            if kind == "fill":
                cv2.rectangle(panel, (x0, y - 12), (x0 + 30, y + 2), tuple(int(0.35 * c + 0.65 * theme["panel"]) for c in color), -1)
                cv2.line(panel, (x0, y - 12), (x0 + 30, y - 12), color, 3, cv2.LINE_AA)
            elif kind == "line":
                cv2.line(panel, (x0, y - 5), (x0 + 30, y - 5), color, 2, cv2.LINE_AA)
            elif kind == "fvo":
                cv2.line(panel, (x0, y - 5), (x0 + 30, y - 5), theme["fvo"], 1, cv2.LINE_AA)
            elif kind == "relative":
                cv2.circle(panel, (x0 + 15, y - 5), 6, RELATIVE_VELOCITY_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(panel, (x0 + 15, y - 5), 6, theme["preferred"], 2, cv2.LINE_AA)
            put_text(panel, text, (x0 + 42, y), 0.5, theme["text"])
            y += 28

        y += 18
        for text in readouts:
            put_text(panel, text, (x0, y), 0.5, theme["text"])
            y += 28

        return panel

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.renderer is not None:
            self.renderer.close()


def set_name(n):
    return "hyperbola" if n == 2 else "super-hyperbola"


def boundary(a, b, n, scale=1.0):
    """
    Boundary v_r = a (1 + (|v_t| / b)^n)^(1 / n) of a collision set, over the
    plotted v_t, for velocities scaled by scale.
    """
    v_t = scale * np.linspace(-PLOT_VT, PLOT_VT, 401)
    v_r = a * (1.0 + (np.abs(v_t) / b) ** n) ** (1.0 / n)
    return v_t, np.minimum(v_r, scale * (PLOT_VR[1] + 0.1))


def to_px(v_t, v_r, scale=1.0):
    """
    Pixels of the plot of the panel, in units of 1/16 px (for cv2 with
    shift=4), for velocities scaled by scale.
    """
    v_t, v_r = np.broadcast_arrays(np.atleast_1d(v_t) / scale, np.atleast_1d(v_r) / scale)
    x = (v_t + PLOT_VT) * PLOT_SCALE
    y = (PLOT_VR[1] - v_r) * PLOT_SCALE
    return np.round(16 * np.c_[x, y]).astype(np.int32)


def put_text(image, text, origin, scale, color):
    cv2.putText(image, text, origin, FONT, scale, color, 1, cv2.LINE_AA)


def skybox_hook(rgb):
    """Prebuild hook of buildModel that adds a flat skybox of colour rgb ("r g b") to the scene."""
    def hook(builder):
        asset = builder.root.find("asset")
        if asset is None:
            asset = ET.SubElement(builder.root, "asset")
        ET.SubElement(asset, "texture", type="skybox", builtin="flat", rgb1=rgb, rgb2=rgb, width="16", height="16")
    return hook


def draw_arrow(scene, origin, vector, color):
    """Arrow from origin along vector, thicker than utils.draw_vector."""
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=[ARROW_WIDTH, ARROW_WIDTH, np.linalg.norm(vector)],
        pos=origin,
        mat=Rz(vector).flatten(),
        rgba=color,
    )
    scene.ngeom += 1


def x264(path):
    """H.264 copy of a video, readable by every player, that replaces the video."""
    output = path.with_name(f"{path.stem}_x264{path.suffix}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", str(output)],
        check=True,
    )
    path.unlink()
    return output


def render(params, other_n, snapshots=(), background="white"):
    label = f"n = {params['n']} ({set_name(params['n'])}), tau = {params['tau']:g} s, layout {params['seed']}"
    if params.get("speed", 1.0) != 1.0:
        label += f", speed x{params['speed']:g}"
    path = VIDEO_DIR / f"crossing_{run_sweep.run_id(params)}.mp4"
    renderer = CrossingRenderer(path, label, params, other_n, snapshots, background)
    skybox = BACKGROUNDS[background]["skybox"]

    try:
        # The controller prints its OSQP status
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_sweep.simulate(params, step_callback=renderer,
                                        prebuild_hook=skybox_hook(skybox) if skybox else None)
    finally:
        renderer.close()

    print(f"{label}: {renderer.frames} frames, completed={result['completed']} at "
          f"{result['time'][-1]:.1f} s, min gap {100 * np.nanmin(result['gap_nn']):+.2f} cm")
    return x264(path), renderer.frames


def side_by_side(videos, output):
    """Videos stacked horizontally, the shorter ones holding their last frame."""
    longest = max(frames for _, frames in videos)
    inputs, filters = [], []
    for index, (path, frames) in enumerate(videos):
        inputs += ["-i", str(path)]
        filters.append(f"[{index}:v]tpad=stop_mode=clone:stop={longest - frames}[v{index}]")
    stack = "".join(f"[v{index}]" for index in range(len(videos)))
    filters.append(f"{stack}hstack=inputs={len(videos)}")

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters),
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output)],
        check=True,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--n", type=int, nargs="+", default=[4])
    parser.add_argument("--tau", type=float, nargs="+", default=[0.5, 5.0])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--snapshots", type=float, nargs="*", default=[], metavar="T")
    parser.add_argument("--background", choices=list(BACKGROUNDS), default="white")
    args = parser.parse_args()

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Same parameters as the crossing study
    base = dict(run_sweep.STUDIES["crossing"][0])
    videos = []
    for n in args.n:
        # The panel compares the collision set with the one of the other degree
        # rendered, or with the hyperbola (the super-hyperbola of n = 4 for n = 2)
        others = [other for other in args.n if other != n]
        other_n = others[0] if len(others) == 1 else (4 if n == 2 else 2)
        for tau in args.tau:
            params = {**base, "n": n, "tau": tau, "seed": args.seed, "speed": args.speed}
            videos.append(render(params, other_n, args.snapshots, args.background))

    for path, _ in videos:
        print(f"saved {path}")

    if args.side_by_side and len(videos) > 1:
        compared = "_".join("_vs_".join(f"{name}{value:g}" for value in values)
                            for name, values in (("n", args.n), ("tau", args.tau)))
        if args.speed != 1.0:
            compared += f"_speed{args.speed:g}"
        output = VIDEO_DIR / f"crossing_seed{args.seed}_{compared}.mp4"
        print(f"saved {side_by_side(videos, output)}")


if __name__ == "__main__":
    main()
