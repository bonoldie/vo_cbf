#!/usr/bin/env python3
"""Desk test: runs crsf_bridge.main() + waypoint_nav against a crude ANGLE-mode quad
behind a fake serial port, in virtual time, with a scripted operator.
Usage: python3 sim_waypoint.py   (needs crsf_bridge.py and waypoint_nav.py alongside)"""
import math
import os
import random
import signal
import struct

import crsf_bridge as br
import waypoint_nav as wn


class Clock:                                   # virtual time for the whole bridge
    t = 1000.0
    def perf_counter(self): return self.t
    def sleep(self, s): self.t += max(s, 1e-4)


clock = Clock()
br.time = clock


class SimQuad:
    def __init__(self, hover_true=1390, wind=(0.8, -0.6), yaw0=10.0, yaw_drift=0.1, fc_yaw_err=-37.0):
        self.lat0, self.lon0 = 45.0, 9.0
        self.n = self.e = self.z = self.vn = self.ve = self.vz = self.roll = self.pitch = 0.0
        self.yaw, self.yaw_drift, self.fc_yaw_err = yaw0, yaw_drift, fc_yaw_err  # true heading, deg
        self.hover, self.wind = hover_true, wind
        self.armed, self.arm_ok, self.ch = False, True, [1500, 1500, 1000, 1500] + [1000] * 12
        self.max_z = self.touch_v = self.max_d = 0.0
        self.ground_t, self.used = 0.0, set()

    def mode(self):
        """Flight-mode string from the user's AUX mapping (ON > 1700, OFF < 1300)."""
        c = self.ch
        if c[7] > 1700:   m = "!FS!"                       # AUX4 failsafe
        elif self.armed and c[9] > 1700: m = "RTH"         # AUX6 GPS rescue
        elif c[8] < 1300: m = "ALTH"                       # AUX5 low  (sim's guess at the string)
        elif c[8] > 1700: m = "POSH"                       # AUX5 high (sim's guess at the string)
        elif c[5] < 1300: m = "STAB"                       # AUX2 low = ANGLE
        elif c[5] > 1700: m = "HOR"
        else:             m = "ACRO"
        return m + ("" if self.armed else "*")

    def step(self, dt):
        c = self.ch
        if c[4] <= 1700:                                   # AUX1 arm switch
            self.armed, self.arm_ok = False, True
        elif (not self.armed and self.arm_ok and c[2] < 1050
              and c[7] < 1700 and c[10] < 1700):           # failsafe / autopilot block arming
            self.armed, self.arm_ok = True, False
        if self.armed:
            self.used |= {n for n, i in (("rescue", 9), ("beeper", 11)) if c[i] > 1700}
            self.used |= {self.mode()}
        if self.armed and c[9] > 1700:                     # crude Betaflight GPS Rescue
            d = math.hypot(self.n, self.e)
            k_v = min(2.0, 0.5 * d) / max(d, 0.1)
            an_c, ae_c = -k_v * self.n - self.vn, -k_v * self.e - self.ve
            yr = math.radians(self.yaw)
            roll_t = max(-15.0, min(15.0, math.degrees(math.atan2(-an_c * math.sin(yr) + ae_c * math.cos(yr), wn.G))))
            pitch_t = max(-15.0, min(15.0, math.degrees(math.atan2(an_c * math.cos(yr) + ae_c * math.sin(yr), wn.G))))
            vz_des = -0.5 if d < 3.0 else (1.0 if self.z < 5.0 else 0.0)
            T, yaw_rate = wn.G * (1.0 + 0.8 * (vz_des - self.vz)), 0.0
            self.ground_t = self.ground_t + dt if (self.z <= 0.0 and vz_des < 0) else 0.0
            if self.ground_t > 1.0:
                self.armed = False                         # landed: Betaflight disarms
        else:
            roll_t, pitch_t = (c[0] - 1500) / 500 * 60, (c[1] - 1500) / 500 * 60
            yaw_rate = (c[3] - 1500) / 500 * 200
            T = (c[2] - 1000) / (self.hover - 1000) * wn.G if self.armed else 0.0
        k = 1 - math.exp(-dt / 0.12)                       # ANGLE mode: tilt follows target
        self.roll += (roll_t - self.roll) * k
        self.pitch += (pitch_t - self.pitch) * k
        if self.z > 0:
            self.yaw = (self.yaw + (yaw_rate + self.yaw_drift) * dt) % 360
        r, p, y = map(math.radians, (self.roll, self.pitch, self.yaw))
        af, ar = T * math.sin(p), T * math.sin(r)
        an = af * math.cos(y) - ar * math.sin(y) - 0.35 * (self.vn - self.wind[0])
        ae = af * math.sin(y) + ar * math.cos(y) - 0.35 * (self.ve - self.wind[1])
        az = T * math.cos(r) * math.cos(p) - wn.G - 0.5 * self.vz
        self.vn += an * dt; self.ve += ae * dt; self.vz += az * dt
        self.z += self.vz * dt
        if self.z <= 0:
            if self.vz < -0.05:
                self.touch_v = max(self.touch_v, -self.vz)
            self.z, self.vz, self.vn, self.ve = 0.0, max(self.vz, 0.0), 0.0, 0.0
        else:
            self.n += self.vn * dt; self.e += self.ve * dt
        self.max_z, self.max_d = max(self.max_z, self.z), max(self.max_d, math.hypot(self.n, self.e))


class FakeSerial:
    """Physics advances on every RC frame written; telemetry comes back 0.3 s late."""
    def __init__(self, *args, **kwargs):
        self.q, self.buf, self.last, self.due, self.pending = SIM, bytearray(), clock.t, {}, []

    @property
    def in_waiting(self):
        while self.pending and self.pending[0][0] <= clock.t:
            self.buf.extend(self.pending.pop(0)[1])
        return len(self.buf)

    def read(self, n):
        out = bytes(self.buf[:n]); del self.buf[:n]
        return out

    def write(self, data):
        bits = int.from_bytes(data[3:25], "little")
        self.q.ch = [(((bits >> (11 * i)) & 0x7FF) - 992) * 5 / 8 + 1500 for i in range(16)]
        dt, self.last = clock.t - self.last, clock.t
        while dt > 1e-9:
            h = min(dt, 0.005); self.q.step(h); dt -= h
        for name, hz in RATES.items():
            if clock.t >= self.due.get(name, 0.0):
                self.due[name] = clock.t + 1.0 / hz
                if not DROP(name, clock.t - T0[0]):
                    self.emit(name)

    def frame(self, typ, payload):
        self.pending.append((clock.t + 0.3, br.build_frame(typ, payload, br.CRSF_ADDR_HANDSET)))

    def emit(self, name):
        q = self.q
        if name == "gps":
            n, e = q.n + random.gauss(0, 0.3), q.e + random.gauss(0, 0.3)
            lat = q.lat0 + math.degrees(n / wn.R_EARTH)
            lon = q.lon0 + math.degrees(e / (wn.R_EARTH * math.cos(math.radians(q.lat0))))
            spd, crs = math.hypot(q.vn, q.ve) * 36, math.degrees(math.atan2(q.ve, q.vn)) % 360 * 100
            alt = int(min(max(q.z, 0), 5000)) + 1000       # Betaflight: whole metres, >= 0
            self.frame(0x02, struct.pack(">iiHHHB", round(lat * 1e7), round(lon * 1e7),
                                         round(spd), round(crs) % 36000, alt, 14))
        elif name == "att":
            yaw = wn.wrap180(q.yaw + q.fc_yaw_err)        # FC yaw frame is arbitrary without a mag
            self.frame(0x1E, struct.pack(">hhh", *(round(math.radians(v) * 10000)
                                                   for v in (-q.pitch, q.roll, yaw))))
        elif name == "mode":
            self.frame(0x21, q.mode().encode() + b"\0")
        elif name == "vario":
            self.frame(0x07, struct.pack(">h", round(q.vz * 100)))
        elif name == "link":
            self.frame(0x14, bytes([196, 196, 100, 10, 0, 7, 3, 190, 100, 8]))
        elif name == "bat":
            self.frame(0x08, struct.pack(">HH", 162, 120) + (900).to_bytes(3, "big") + bytes([70]))

    def close(self):
        pass


class Operator:
    """Scripted keyboard; 'SIGINT' sends a real Ctrl+C to the process."""
    def __init__(self, script):
        self.script, self.flew, self.timed_out = sorted(script), False, False

    def empty(self):
        nav, el = NAV[0], clock.t - T0[0]
        self.flew |= nav.airborne
        if self.flew and nav.state == "DISARMED" and not self.script and not nav.quit:
            self.script.append((0.0, "q"))
        if el > 300 and not self.timed_out:
            self.timed_out, self.script[:0] = True, [(0.0, "k"), (0.0, "q")]
        while self.script and self.script[0][1] == "SIGINT" and el >= self.script[0][0]:
            self.script.pop(0)
            print(">>> operator: Ctrl+C")
            os.kill(os.getpid(), signal.SIGINT)
        return not (self.script and el >= self.script[0][0])

    def get_nowait(self):
        cmd = self.script.pop(0)[1]
        print(f">>> operator: {cmd}")
        return cmd


NAV, T0 = [], [0.0]


class SimNav(wn.WaypointNav):
    def __init__(self, t):
        super().__init__(t)
        NAV.append(self)


def run(title, script, drop=lambda name, el: False, rates=None, **quad):
    global SIM, DROP, RATES
    print(f"\n===== {title} =====")
    SIM, DROP, T0[0] = SimQuad(**quad), drop, clock.t
    RATES = rates or {"gps": 5, "att": 10, "mode": 2, "vario": 5, "link": 4, "bat": 1}
    NAV.clear()
    br.serial.Serial, br.WaypointNav = FakeSerial, SimNav
    br.keyboard_commands = lambda: Operator(script)
    br.print_status, br.STATUS_PRINT_S, br.HELP = (lambda t: None), 2.0, ""
    br.main()
    q = SIM
    print(f"RESULT: on ground at N{q.n:+.1f} E{q.e:+.1f}, max alt {q.max_z:.1f} m, max dist "
          f"{q.max_d:.1f} m, touchdown {q.touch_v:.2f} m/s, sim time {clock.t - T0[0]:.0f} s, "
          f"{'ARMED' if q.armed else 'disarmed'}; FC modes/switches seen armed: {sorted(q.used)}")


SCENARIOS = {
    "1": ("no compass: nose 10 deg off the WP1 bearing, gyro drift, wind, hover 40 us off", {}),
    "2": ("same, but the FC sends no vario",
          {"rates": {"gps": 5, "att": 10, "mode": 2, "link": 4, "bat": 1}}),
    "3": ("HAS_MAG=True: nose starts pointing east, FC yaw = true heading",
          {"mag": True, "yaw0": 90.0, "fc_yaw_err": 0.0}),
    "4": ("no compass, nose pointed 120 deg wrong", {"yaw0": 120.0}),
    "5": ("GPS telemetry lost for 2.8 s: returns while blind, script lands",
          {"drop": lambda name, el: name == "gps" and 20 < el < 22.8}),
    "9": ("ALL telemetry lost for 15 s: Betaflight GPS Rescue brings it home",
          {"drop": lambda name, el: 20 < el < 35}),
    "10": ("'r' during leg 1: Betaflight GPS Rescue flies home, lands, disarms",
           {"extra": [(19, "r")]}),
    "11": ("'r' then 'l': rescue cancelled, the script lands where it is",
           {"extra": [(19, "r"), (23, "l")]}),
    "6": ("Ctrl+C in the air (lands), then Ctrl+C on the ground (quits)",
          {"extra": [(18, "SIGINT"), (45, "SIGINT")]}),
    "8": ("Ctrl+C twice in the air = KILL", {"extra": [(18, "SIGINT"), (19, "SIGINT"), (25, "q")]}),
    "7": ("'y' before the checks pass, then a proper arm and a disarm on the ground",
          {"script": [(0.5, "a"), (0.6, "y"), (4, "y"), (7, "d"), (8, "q")]}),
}

if __name__ == "__main__":
    import sys
    random.seed(1)
    for key in (sys.argv[1:] or SCENARIOS):
        title, opt = SCENARIOS[key]
        opt = dict(opt)
        wn.HAS_MAG = opt.pop("mag", False)
        script = opt.pop("script", [(2, "a"), (6, "y"), (9, "g")] + opt.pop("extra", []))
        run(f"{key}: {title}", script, **opt)
