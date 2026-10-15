#!/usr/bin/env python3
"""
waypoint_nav.py - GPS waypoint autopilot for the CRSF bridge (Betaflight ANGLE mode)

    nav = WaypointNav(t)          # t = the bridge's Telemetry object
    channels = nav.update(now)    # every RC tick, instead of a fixed channel list

Position -> velocity -> tilt drives the roll/pitch sticks; altitude -> climb rate
-> throttle drives the throttle. Betaflight's ANGLE mode turns the sticks into
attitude, so the PC only closes the slow outer loops on GPS telemetry.
"""

import math
import queue
import sys
import threading
import traceback

HELP = """Commands (letter + Enter):
  a  arm stage 1: pre-arm checks          y  arm stage 2: arm switch ON
  g  go: take off and fly the waypoints   h  hold position ('g' resumes)
  l  land here (also cancels a rescue)    r  Betaflight GPS Rescue: home and land
  k  KILL: disarm now, even in the air    d  disarm (ground)
  b  bench test while disarmed            q  quit (ground)
  Ctrl+C: quit on the ground; in the air land, twice = KILL"""

# ---- Mission ----------------------------------------------------------------
# (north_m, east_m, alt_m) from the take-off point, or ("ll", lat, lon, alt_m).
# The drone lands at the last waypoint.
WAYPOINTS = [(8.0, 8.0, 4.0), (0.0, 0.0, 4.0)]
TAKEOFF_ALT_M, WP_RADIUS_M, WP_HOLD_S = 3.0, 2.0, 3.0
# False (no compass): point the nose at WP1 before arming. The script never yaws
#        and uses that bearing as the heading for the whole flight.
# True: Betaflight's yaw is a true heading (calibrated mag, `set trust_mag = ON` on 2026.6,
#       checked against a phone compass with 'b'); the nose turns to the target.
HAS_MAG = False

# ---- Aircraft / Betaflight ----------------------------------------------------
HOVER_US = 1350             # ANGLE-mode hover throttle: MEASURE IT
ANGLE_LIMIT_DEG = 60.0      # Betaflight angle_limit (tilt at full stick)
BATT_LAND_V = 0.0           # land below this pack voltage (e.g. 14.0 for 4S), 0 = off


# ---- Your Betaflight Modes tab (ON > 1700, OFF < 1300) --------------------------
def aux(n):
    """Channel index of AUX n (AUX1 = CH5 = index 4)."""
    return n + 3


ARM_CH = aux(1)             # ON = armed
RESCUE_CH = aux(6)          # GPS Rescue: command 'r' and the lost-telemetry fallback
                            # (None = never touch it, use the blind descent instead)
BEEPER_CH = aux(8)          # beeps while blind or rescuing, to help find the drone (None = off)
AUX_DEFAULTS = {
    aux(2): 1000,           # < 1300 = ANGLE (> 1700 would be HORIZON)
    aux(3): 1000,           # MAG heading lock OFF: the script handles yaw itself
    aux(4): 1000,           # FAILSAFE OFF (ON = manual stage-2 failsafe)
    aux(5): 1500,           # middle = neither ALTHOLD (< 1300) nor POSHOLD (> 1700)
    aux(6): 1000,           # GPS RESCUE OFF unless the script turns it on
    aux(7): 1000,           # AUTOPILOT OFF: SITL-only in 2026.6, and it blocks arming
    aux(8): 1000,           # BEEPER OFF
    aux(9): 1000,           # VTX PIT MODE OFF
}                           # AUX10..AUX12 (CH14..CH16) unused, sent as 1000

# ---- Gains and limits -----------------------------------------------------------
V_MAX, KP_POS = 3.0, 0.4                    # m/s, 1/s: brakes from V_MAX/KP_POS m out
KV, KI_V, I_V_MAX = 1.0, 0.15, 1.0          # velocity loop + wind integrator (m/s^2)
TILT_MAX_DEG = 15.0
K_YAW, YAW_MAX_US = 3.0, 80.0               # yaw stick us per degree (HAS_MAG only)
KP_Z, VZ_UP, VZ_DOWN = 0.5, 1.0, 0.6        # altitude -> climb rate (m/s)
KP_VZ, KI_VZ, I_VZ_MAX = 20.0, 8.0, 200.0   # climb rate -> throttle (us)
THR_MIN, THR_MAX = HOVER_US - 150, HOVER_US + 250
SLEW = (250.0, 250.0, 300.0, 300.0)         # us/s: roll, pitch, throttle, yaw

# ---- Safety -----------------------------------------------------------------------
MIN_SATS, LQ_ARM, LQ_MIN = 6, 90, 60
MAX_DIST_M, MAX_ALT_M, FLYAWAY_M = 50.0, 15.0, 8.0
GPS_STALE_S, ATT_STALE_S, MODE_STALE_S = 1.5, 1.0, 3.0   # ~3x your telemetry intervals
BLIND_S = 10.0              # no telemetry and no RESCUE_CH: level, below hover, then disarm
RESCUE_AFTER_S = 2.0        # no telemetry: wait this long for it to return, then GPS Rescue

# Telemetry frame types; the bridge stamps their arrival time in t.rx_time
GPS, VARIO, BATT, LINK, ATT, MODE = 0x02, 0x07, 0x08, 0x14, 0x1E, 0x21
FIELD = {GPS: "gps", VARIO: "vario", BATT: "battery", LINK: "link",
         ATT: "attitude", MODE: "flight_mode"}
FLYING = ("TAKEOFF", "NAV", "HOLD", "LAND", "BLIND", "FC_FS")
G, R_EARTH = 9.81, 6371000.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def ll_to_ne(lat, lon, lat0, lon0):
    """Flat-earth offset in metres, fine over a few km."""
    return (math.radians(lat - lat0) * R_EARTH,
            math.radians(wrap180(lon - lon0)) * R_EARTH * math.cos(math.radians(lat0)))


def safe_channels():
    """Disarmed, sticks centred, throttle low, every mode switch at its default."""
    ch = [1500, 1500, 1000, 1500] + [1000] * 12
    for i, v in AUX_DEFAULTS.items():
        ch[i] = v
    return ch


def keyboard_commands():
    """Lines typed on stdin, collected by a background thread."""
    q = queue.Queue()

    def reader():
        for line in sys.stdin:
            q.put(line.strip().lower())

    threading.Thread(target=reader, daemon=True).start()
    return q


class WaypointNav:
    def __init__(self, telemetry):
        self.t = telemetry
        self.state, self.state_t, self.now, self.last = "DISARMED", 0.0, 0.0, None
        self.quit = self.arm = self.ready = self.ctrl_c_land = self.recover = False
        self.rescue = False                             # our GPS Rescue switch
        self.timers, self.nag_t = {}, 0.0
        self.home, self.targets, self.leg = None, [], 0
        self.hdg0 = self.hdg = 0.0
        self.alt_zero = self.z = self.vz = 0.0          # altitude above take-off, m, m/s
        self.z_t = self.gps_seen = self.vario_seen = None
        self.i_n = self.i_e = self.i_vz = 0.0
        self.roll = self.pitch = self.yaw = 0.0         # deg (+ right / + nose down), yaw us
        self.thr, self.hover_ref = 1000.0, float(HOVER_US)
        self.out = [1500.0, 1500.0, 1000.0, 1500.0]
        self.dist = self.min_dist = 0.0
        self.arrived = self.final = self.land_pt = self.hold_pt = None

    @property
    def airborne(self):
        return self.state in FLYING

    # ---- helpers ---------------------------------------------------------------
    def age(self, frame):
        """Seconds since that telemetry frame was decoded (inf if never)."""
        if getattr(self.t, FIELD[frame], None) is None:
            return math.inf
        return self.now - self.t.rx_time.get(frame, -math.inf)

    def nav_ok(self):
        return self.age(GPS) < GPS_STALE_S and self.age(ATT) < ATT_STALE_S

    def fc_mode(self):
        return (self.t.flight_mode or "").rstrip("*")

    def fc_armed(self):                       # Betaflight appends "*" while disarmed
        return self.t.flight_mode is not None and not self.t.flight_mode.endswith("*")

    def pos_ne(self):
        return ll_to_ne(self.t.gps[0], self.t.gps[1], *self.home)

    def set_state(self, state, msg=""):
        print(f"[{self.state} -> {state}] {msg}")
        self.state, self.state_t = state, self.now
        self.timers.clear()

    def held(self, name, cond, seconds):
        """True once `cond` has been true without a break for `seconds`."""
        if not cond:
            self.timers.pop(name, None)
            return False
        return self.now - self.timers.setdefault(name, self.now) >= seconds

    def level(self, thr):
        self.roll = self.pitch = self.yaw = 0.0
        self.thr = thr

    def disarm(self, msg):
        self.arm = self.ctrl_c_land = self.rescue = False
        self.level(1000.0)
        self.set_state("DISARMED", msg)

    # ---- operator ----------------------------------------------------------------
    def command(self, cmd, now):
        self.now = now
        try:
            self._command(cmd)
        except Exception:                     # a command must never stop the RC loop
            traceback.print_exc()

    def _command(self, cmd):
        st = self.state
        if cmd == "k":
            self.disarm("KILL")
        elif cmd == "ctrl_c":
            if not self.airborne:
                self.quit = True
            elif self.ctrl_c_land:
                self.disarm("Ctrl+C twice: KILL")
            else:
                self.ctrl_c_land = True
                if st in ("TAKEOFF", "NAV", "HOLD"):
                    self.land(True, "Ctrl+C: landing (Ctrl+C again = KILL)")
                else:
                    print("[Ctrl+C] automatic descent or rescue running (Ctrl+C again = KILL)")
        elif cmd == "a" and st in ("DISARMED", "BENCH"):
            self.ready, self.nag_t = False, 0.0
            self.set_state("PREARM", "ARM 1/2: pre-arm checks running")
        elif cmd == "y" and st == "PREARM" and self.ready:
            self.arm = True
            self.set_state("ARMING", "ARM 2/2: arm switch ON, waiting for the FC")
        elif cmd == "g" and st == "ARMED":
            self.takeoff()
        elif cmd == "g" and st == "HOLD":
            self.start_leg(self.leg)
        elif cmd == "h" and st in ("TAKEOFF", "NAV", "LAND"):
            self.hold_pt = (*self.pos_ne(), max(self.z, 2.0))
            self.set_state("HOLD", "operator: 'g' resumes, 'l' lands")
        elif cmd == "l" and st in ("TAKEOFF", "NAV", "HOLD"):
            self.land(True, "operator")
        elif cmd == "l" and st == "FC_FS" and self.rescue:
            self.rescue = False
            print("[FC_FS] rescue switch OFF: the script lands once Betaflight is back in ANGLE")
        elif cmd == "r" and st in ("TAKEOFF", "NAV", "HOLD", "LAND", "BLIND") and RESCUE_CH is not None:
            self.start_rescue("operator")
        elif cmd == "d" and st in ("BENCH", "PREARM", "ARMING", "ARMED"):
            self.disarm("operator")
        elif cmd == "b" and st == "DISARMED":
            self.bench()
        elif cmd == "q" and not self.airborne:
            self.quit = True
        elif cmd:
            why = ": checks not OK yet" if cmd == "y" and st == "PREARM" else ""
            print(f"[cmd] '{cmd}' ignored in {st}{why}")

    # ---- main step: once per RC frame ----------------------------------------------
    def update(self, now):
        """Returns the 16 channel values (us) to send now."""
        dt = 0.0 if self.last is None else clamp(now - self.last, 0.0, 0.1)
        self.now = self.last = now
        try:
            if self.home is not None:
                self.estimate(dt)
            if self.airborne:
                self.safety_checks()
            getattr(self, "st_" + self.state.lower())(dt)
        except Exception as e:                # a bug must not stop the RC stream mid-air
            if self.state not in ("BLIND", "FC_FS", "DISARMED"):
                traceback.print_exc()
                if self.airborne:
                    self.blind(f"software error {e!r}", recover=False)
                else:
                    self.disarm(f"software error {e!r}")
            if self.state == "BLIND":
                self.st_blind(dt)
            elif self.state == "FC_FS":
                self.level(self.hover_ref - 20.0)
        cmd = self.channels(dt)
        return cmd

    def estimate(self, dt):
        """Heading, plus an alpha-beta filter on the 1 m altitude steps (+ vario)."""
        t = self.t
        self.hdg = t.attitude[2] % 360.0 if HAS_MAG else self.hdg0
        self.z += self.vz * dt
        stamp = t.rx_time.get(GPS)
        if stamp != self.gps_seen:
            self.gps_seen = stamp
            r = t.gps[4] - self.alt_zero - self.z
            self.z += 0.4 * r
            if self.z_t is not None:
                self.vz += 0.1 * r / max(stamp - self.z_t, 0.05)
            self.z_t = stamp
        stamp = t.rx_time.get(VARIO)
        if stamp != self.vario_seen and self.age(VARIO) < 1.0:
            self.vario_seen = stamp
            self.vz += 0.5 * (t.vario - self.vz)

    def safety_checks(self):
        t, st = self.t, self.state
        mode_ok = self.age(MODE) < MODE_STALE_S
        if mode_ok and not self.fc_armed():
            self.disarm(f"FC reports disarmed ({t.flight_mode})")
        elif mode_ok and self.fc_mode() in ("!FS!", "RTH"):
            if st != "FC_FS":
                self.hover_ref = HOVER_US + self.i_vz
                self.set_state("FC_FS", f"Betaflight failsafe ({self.fc_mode()}) flying, 'k' = KILL")
        elif st in ("BLIND", "FC_FS"):
            pass
        elif mode_ok and self.fc_mode() != "STAB":
            self.blind(f"FC left ANGLE mode ({self.fc_mode()})", recover=False)
        elif not self.nav_ok():
            self.blind("GPS/attitude telemetry stale", recover=True)
        elif st == "LAND":
            pass
        elif self.held("lq", self.age(LINK) > 2.0 or t.link["lq"] < LQ_MIN, 2.0):
            self.land(True, "link quality low")
        elif BATT_LAND_V and self.held("batt", self.age(BATT) < 5.0 and t.battery[0] < BATT_LAND_V, 3.0):
            self.land(True, "battery low")
        elif math.hypot(*self.pos_ne()) > MAX_DIST_M or self.z > MAX_ALT_M:
            self.land(False, "geofence: landing level")

    # ---- two-stage arming ------------------------------------------------------------
    def make_targets(self, lat0, lon0):
        return [(*ll_to_ne(w[1], w[2], lat0, lon0), w[3]) if w[0] == "ll" else tuple(w)
                for w in WAYPOINTS]

    def prearm_failures(self):
        t, f = self.t, []
        if self.age(LINK) > 2.0 or t.link["lq"] < LQ_ARM:
            f.append(f"link LQ below {LQ_ARM}%")
        if self.age(GPS) > GPS_STALE_S or t.gps[5] < MIN_SATS:
            f.append(f"GPS stale or fewer than {MIN_SATS} sats")
        if self.age(ATT) > ATT_STALE_S or max(abs(t.attitude[0]), abs(t.attitude[1])) > 15:
            f.append("attitude stale or not level")
        mode = self.fc_mode()
        if self.age(MODE) > MODE_STALE_S:
            f.append("no flight-mode telemetry")
        elif self.fc_armed():
            f.append("FC already armed")
        elif mode == "!ERR":
            f.append("FC arming disabled: check the arming flags in Betaflight")
        elif mode not in ("STAB", "WAIT", "ANGL"):    # WAIT = GPS home not set yet, normal
            f.append(f"FC mode {mode}: ANGLE not active")
        if BATT_LAND_V and (self.age(BATT) > 5.0 or t.battery[0] < BATT_LAND_V + 1.0):
            f.append("battery low or no battery telemetry")
        if not f:
            tg = self.make_targets(t.gps[0], t.gps[1])
            f += [f"WP{i + 1} outside the geofence/altitude limits" for i, (n, e, z) in enumerate(tg)
                  if math.hypot(n, e) > MAX_DIST_M - 5 or not 2.0 <= z <= MAX_ALT_M - 1]
            if not tg or (not HAS_MAG and math.hypot(*tg[0][:2]) < 5.0):
                f.append("WP1 must be at least 5 m away (the nose points at it)")
        return f

    def st_prearm(self, dt):
        fails = self.prearm_failures()
        ready = self.held("ok", not fails, 1.0)
        if ready and not self.ready:
            hint = ""
            if not HAS_MAG:
                n, e, _ = self.make_targets(self.t.gps[0], self.t.gps[1])[0]
                hint = f" Nose must point at WP1, bearing {math.degrees(math.atan2(e, n)) % 360:03.0f}°."
            print(f"[ARM 1/2] checks OK.{hint} Type 'y' to arm.")
        self.ready = ready
        if fails and self.now >= self.nag_t:
            print("[ARM 1/2] waiting: " + "; ".join(fails))
            self.nag_t = self.now + 2.0
        if self.now - self.state_t > 30.0:
            self.disarm("pre-arm timed out")

    def st_arming(self, dt):
        self.level(1000.0)
        if self.t.rx_time.get(MODE, -math.inf) > self.state_t and self.fc_armed():
            if self.fc_mode() in ("STAB", "ANGL"):
                self.set_state("ARMED", "FC armed in ANGLE. 'g' = take off, 'd' = disarm")
            else:
                self.disarm(f"FC armed in {self.fc_mode()}, not ANGLE")
        elif self.now - self.state_t > 4.0:
            self.disarm(f"FC did not arm (mode {self.t.flight_mode}): check arming flags")

    def st_armed(self, dt):
        self.level(1000.0)
        if self.age(MODE) < MODE_STALE_S and not self.fc_armed():
            self.disarm(f"FC disarmed ({self.t.flight_mode})")
        elif self.now - self.state_t > 20.0:
            self.disarm("no 'g' within 20 s")

    def st_disarmed(self, dt):
        self.level(1000.0)

    # ---- motion ------------------------------------------------------------------------
    def capture_home(self):
        t = self.t
        self.home = (t.gps[0], t.gps[1])
        self.targets = self.make_targets(*self.home)
        n, e, _ = self.targets[0]
        self.hdg0 = math.degrees(math.atan2(e, n)) % 360.0     # the nose points at WP1
        self.alt_zero, self.z, self.vz, self.z_t = t.gps[4], 0.0, 0.0, None
        self.gps_seen = t.rx_time.get(GPS)
        self.i_n = self.i_e = self.i_vz = 0.0
        self.estimate(0.0)

    def takeoff(self):
        if not (self.fc_armed() and self.fc_mode() in ("STAB", "ANGL") and self.nav_ok()):
            print("[GO] refused: needs the FC armed in ANGLE and fresh GPS/attitude")
            return
        self.capture_home()
        self.i_vz = -40.0                     # spool to just below the hover guess, then search
        self.set_state("TAKEOFF", f"spool-up, then climb to {TAKEOFF_ALT_M:.1f} m")

    def st_takeoff(self, dt):
        el = self.now - self.state_t
        if el < 2.0:                                      # spool up on the ground
            self.level(1000.0 + (HOVER_US + self.i_vz - 1000.0) * el / 2.0)
            return
        z_ref = min(TAKEOFF_ALT_M, 0.7 * (el - 2.0))      # climb ramp at 0.7 m/s
        self.roll = self.pitch = self.yaw = 0.0
        if self.z < 0.3 and self.vz < 0.2:                # not climbing yet: search for hover
            self.i_vz = min(self.i_vz + 15.0 * dt, I_VZ_MAX)
        self.thr = self.alt_control(dt, z_ref, 0.7 if z_ref < TAKEOFF_ALT_M else 0.0)
        if self.z > TAKEOFF_ALT_M - 0.5:
            self.start_leg(0)
        elif el > 15.0:                                   # never cut the motors here: land
            self.land(False, "take-off too slow: check HOVER_US")

    def start_leg(self, i):
        self.leg, self.arrived = i, None
        n, e, z = self.targets[i]
        pn, pe = self.pos_ne()
        self.min_dist = math.hypot(n - pn, e - pe)
        self.set_state("NAV", f"leg {i + 1}/{len(self.targets)}: N{n:+.1f} E{e:+.1f} "
                              f"alt {z:.1f} m, {self.min_dist:.1f} m away")

    def st_nav(self, dt):
        n, e, z = self.targets[self.leg]
        d = self.xy_control(dt, n, e, face_target=True)
        self.thr = self.alt_control(dt, z)
        self.min_dist = min(self.min_dist, d)
        if d > self.min_dist + FLYAWAY_M:
            self.land(False, "moving AWAY from the target (heading wrong?): landing level")
        elif d < WP_RADIUS_M and self.t.gps[2] < 3.6 and abs(z - self.z) < 1.0:
            if self.arrived is None:
                self.arrived = self.now
                print(f"[NAV] at WP{self.leg + 1}, holding {WP_HOLD_S:.0f} s")
            elif self.now - self.arrived > WP_HOLD_S:
                if self.leg + 1 < len(self.targets):
                    self.start_leg(self.leg + 1)
                else:
                    self.land(True, "mission complete")
        elif d > 1.5 * WP_RADIUS_M:
            self.arrived = None

    def st_hold(self, dt):
        n, e, z = self.hold_pt
        self.xy_control(dt, n, e)
        self.thr = self.alt_control(dt, z)

    def land(self, hold_xy, msg):
        self.land_pt = self.pos_ne() if hold_xy and self.age(GPS) < GPS_STALE_S else None
        self.hover_ref, self.final = HOVER_US + self.i_vz, None
        self.set_state("LAND", msg)

    def st_land(self, dt):
        if self.final is None and self.z < 1.0:
            self.final = self.now                         # last metre: level, push down
        if self.land_pt and self.final is None:
            self.xy_control(dt, *self.land_pt)
        else:
            self.roll = self.pitch = self.yaw = 0.0
        if self.final is not None:                        # so touchdown shows in the throttle
            self.i_vz = max(self.i_vz - 8.0 * dt, -I_VZ_MAX)
        self.thr = self.alt_control(dt, vz_ref=-0.5 if self.z > 2.5 else -0.3)
        down = self.final is not None and self.thr <= max(self.hover_ref - 60.0, THR_MIN)
        if self.held("down", down, 1.0) or (self.final is not None and self.now - self.final > 8.0):
            self.disarm("landed")

    def blind(self, msg, recover):
        self.hover_ref, self.recover = HOVER_US + self.i_vz, recover
        then = (f"GPS Rescue in {RESCUE_AFTER_S:.0f} s" if RESCUE_CH is not None
                else f"disarm in {BLIND_S:.0f} s")
        self.set_state("BLIND", f"{msg}: level, throttle {self.hover_ref - 20:.0f}, {then}")

    def st_blind(self, dt):
        self.level(self.hover_ref - 20.0)
        if self.recover and self.nav_ok():
            self.z, self.z_t = self.t.gps[4] - self.alt_zero, None
            self.land(True, "telemetry back: landing")
        elif RESCUE_CH is not None and self.now - self.state_t > RESCUE_AFTER_S:
            self.start_rescue("still blind")
        elif self.now - self.state_t > BLIND_S:
            self.disarm("blind descent over")

    def start_rescue(self, msg):
        self.rescue, self.hover_ref = True, HOVER_US + self.i_vz
        self.set_state("FC_FS", f"{msg}: Betaflight GPS Rescue ON. 'l' = cancel and land, 'k' = KILL")

    def st_fc_fs(self, dt):
        # Betaflight ignores these sticks while its rescue/failsafe flies. They only
        # matter if it isn't flying, and then a blind drone should come down slowly.
        self.level(self.hover_ref - (0.0 if self.nav_ok() else 20.0))
        back = self.fc_mode() in ("STAB", "ANGL") and self.nav_ok() and self.age(MODE) < MODE_STALE_S
        if self.held("back", back, 3.0 if self.rescue else 0.5):
            if self.rescue:
                print("[FC_FS] Betaflight did not enter GPS Rescue: the script lands instead")
            self.rescue = False
            self.z, self.z_t = self.t.gps[4] - self.alt_zero, None
            self.land(True, "Betaflight in ANGLE again: landing")

    def bench(self):
        if not self.nav_ok():
            print("[BENCH] needs fresh GPS and attitude telemetry")
            return
        self.capture_home()
        self.set_state("BENCH", "disarmed: carry the drone around, watch 'cmd'; 'd' stops")

    def st_bench(self, dt):
        if self.age(GPS) < GPS_STALE_S:
            self.xy_control(dt, *self.targets[0][:2], face_target=True)

    # ---- controllers ---------------------------------------------------------------------
    def xy_control(self, dt, tn, te, face_target=False):
        """Position error -> desired velocity -> acceleration -> ANGLE-mode tilt."""
        t = self.t
        n, e = self.pos_ne()
        spd, crs = t.gps[2] / 3.6, math.radians(t.gps[3])
        vn, ve = spd * math.cos(crs), spd * math.sin(crs)
        dn, de = tn - n, te - e
        d = self.dist = math.hypot(dn, de)
        k = min(V_MAX, KP_POS * d) / max(d, 0.01)         # desired speed along the error
        en, ee = k * dn - vn, k * de - ve                  # velocity error, m/s
        an, ae = KV * en + self.i_n, KV * ee + self.i_e
        a_max = G * math.tan(math.radians(TILT_MAX_DEG))
        a = math.hypot(an, ae)
        if a > a_max:                                      # cap the tilt, keep the direction
            an, ae = an * a_max / a, ae * a_max / a
        elif d < 6.0:                                      # wind integrator near the target
            self.i_n = clamp(self.i_n + KI_V * en * dt, -I_V_MAX, I_V_MAX)
            self.i_e = clamp(self.i_e + KI_V * ee * dt, -I_V_MAX, I_V_MAX)
        psi = math.radians(self.hdg)
        fwd = an * math.cos(psi) + ae * math.sin(psi)      # rotate into the body frame
        right = -an * math.sin(psi) + ae * math.cos(psi)
        self.pitch = math.degrees(math.atan2(fwd, G))      # + = nose down = forward
        self.roll = math.degrees(math.atan2(right, G))     # + = roll right
        self.yaw = 0.0
        if HAS_MAG and face_target and d > 5.0:
            err = wrap180(math.degrees(math.atan2(de, dn)) - self.hdg)
            self.yaw = clamp(K_YAW * err, -YAW_MAX_US, YAW_MAX_US)
        return d

    def alt_control(self, dt, z_ref=None, vz_ff=0.0, vz_ref=None):
        """Altitude -> climb rate -> throttle; i_vz learns the real hover throttle."""
        if vz_ref is None:
            vz_ref = clamp(KP_Z * (z_ref - self.z) + vz_ff, -VZ_DOWN, VZ_UP)
        err = vz_ref - self.vz
        tilt = math.radians(math.hypot(self.roll, self.pitch))
        base = 1000.0 + (HOVER_US - 1000.0) / max(math.cos(tilt), 0.85)
        i = clamp(self.i_vz + KI_VZ * err * dt, -I_VZ_MAX, I_VZ_MAX)
        thr = base + KP_VZ * err + i
        if THR_MIN < thr < THR_MAX:                        # anti-windup
            self.i_vz = i
        return clamp(thr, THR_MIN, THR_MAX)

    # ---- output ----------------------------------------------------------------------------
    def channels(self, dt):
        ch = safe_channels()
        if not self.arm:                                   # disarmed: snap to safe values
            self.out = [1500.0, 1500.0, 1000.0, 1500.0]
            return ch
        want = (1500 + 500 * clamp(self.roll / ANGLE_LIMIT_DEG, -1, 1),
                1500 + 500 * clamp(self.pitch / ANGLE_LIMIT_DEG, -1, 1),
                self.thr, 1500 + self.yaw)
        self.out = [o + clamp(w - o, -s * dt, s * dt) for o, w, s in zip(self.out, want, SLEW)]
        ch[:4] = [round(v) for v in self.out]              # armed: slew-limited sticks
        ch[ARM_CH] = 2000
        if RESCUE_CH is not None and self.rescue:
            ch[RESCUE_CH] = 2000
        if BEEPER_CH is not None and self.state in ("BLIND", "FC_FS"):
            ch[BEEPER_CH] = 2000
        return ch

    def status(self):
        s = f"[{self.state}" + (f" {self.leg + 1}/{len(self.targets)}" if self.state == "NAV" else "") + "]"
        if self.home and self.t.gps:
            n, e = self.pos_ne()
            s += f" N{n:+.1f} E{e:+.1f} hdg {self.hdg:03.0f}°"
            if self.state in ("NAV", "BENCH"):
                s += f" dist {self.dist:.1f}m"
        if self.airborne:
            s += f" z {self.z:.1f} vz {self.vz:+.1f} thr {self.thr:.0f} hover~{HOVER_US + self.i_vz:.0f}"
        if self.state in ("BENCH", "TAKEOFF", "NAV", "HOLD", "LAND"):
            s += f" | cmd R{self.roll:+.1f}° P{self.pitch:+.1f}° Y{self.yaw:+.0f}us"
        return s
