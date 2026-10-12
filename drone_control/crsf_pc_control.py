#!/usr/bin/env python3
"""
PC -> serial bridge -> ExpressLRS TX module -> ELRS RX -> Betaflight

Sends CRSF RC_CHANNELS_PACKED frames and decodes the telemetry coming back.
Channel order is AETR (Betaflight default map):
    CH1 Roll, CH2 Pitch, CH3 Throttle, CH4 Yaw, CH5 AUX1 (arm), CH6..CH16 AUX2..AUX12

TEST WITH PROPS OFF.
"""

import math
import struct
import time

import serial

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PORT = "/dev/ttyACM1"
BAUD = 115200          # Only matters for real UARTs / ST-Link VCP. For native USB CDC
                       # the STM32's UART setting toward the TX module is what counts.
DEFAULT_RATE_HZ = 100  # Used until the TX module reports its packet rate
MAX_RATE_HZ = 250      # Cap: a 26-byte frame takes ~2.26 ms at 115200 baud
STATUS_PRINT_S = 0.5   # Telemetry summary print interval

# --------------------------------------------------------------------------
# CRSF constants
# --------------------------------------------------------------------------
CRSF_SYNC = 0xC8              # Sync byte / flight controller address
CRSF_ADDR_HANDSET = 0xEA      # "Radio transmitter" = this script
CRSF_ADDR_TX_MODULE = 0xEE    # "CRSF transmitter"  = the ELRS TX module
VALID_START_BYTES = {CRSF_SYNC, CRSF_ADDR_HANDSET, CRSF_ADDR_TX_MODULE}

FRAME_BATTERY = 0x08
FRAME_LINK_STATS = 0x14
FRAME_RC_CHANNELS = 0x16
FRAME_ATTITUDE = 0x1E
FRAME_GPS = 0x02
FRAME_FLIGHT_MODE = 0x21

# ELRS handset timing sync lives here (subtype 0x10)
FRAME_RADIO_ID = 0x3A

CRSF_MAX_LEN_FIELD = 62       # Max frame = 64 bytes: addr + len + 62
CRSF_CH_MIN, CRSF_CH_MAX = 172, 1811


def crc8_dvb_s2(data, crc=0):
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


# --------------------------------------------------------------------------
# Outgoing frames
# --------------------------------------------------------------------------
def us_to_crsf(us):
    """Exact inverse of Betaflight's mapping: us = (crsf - 992) * 5/8 + 1500."""
    value = round((us - 1500) * 8 / 5 + 992)
    return max(CRSF_CH_MIN, min(CRSF_CH_MAX, value))


def pack_channels(channels_us):
    """16 channels x 11 bits, LSB first -> 22 bytes."""
    if len(channels_us) != 16:
        raise ValueError("CRSF RC frame needs exactly 16 channels")
    bits = 0
    for i, us in enumerate(channels_us):
        bits |= us_to_crsf(us) << (11 * i)
    return bits.to_bytes(22, "little")


def build_frame(frame_type, payload, address=CRSF_ADDR_TX_MODULE):
    body = bytes([frame_type]) + bytes(payload)  # CRC covers type + payload
    return bytes([address, len(body) + 1]) + body + bytes([crc8_dvb_s2(body)])


def rc_frame(channels_us):
    return build_frame(FRAME_RC_CHANNELS, pack_channels(channels_us))


# --------------------------------------------------------------------------
# Incoming telemetry
# --------------------------------------------------------------------------
class Telemetry:
    def __init__(self):
        self.battery = None       # (volts, amps, mAh used, % remaining)
        self.attitude = None      # (pitch, roll, yaw) in degrees
        self.gps = None           # (lat, lon, speed km/h, heading deg, alt m, sats)
        self.link = None          # dict with uplink/downlink stats
        self.flight_mode = None   # string from Betaflight, e.g. "ACRO", "!FS!"
        self.packet_period = 1.0 / DEFAULT_RATE_HZ

def rssi_dbm(raw):
    """ELRS sends signed negative int8; TBS Crossfire sends positive magnitude."""
    return raw - 256 if raw >= 128 else -raw

def handle_frame(frame_type, p, t):
    try:
        if frame_type == FRAME_BATTERY and len(p) >= 8:
            volt, curr = struct.unpack_from(">HH", p, 0)       # both in 0.1 units
            used_mah = int.from_bytes(p[4:7], "big")
            t.battery = (volt / 10, curr / 10, used_mah, p[7])

        elif frame_type == FRAME_ATTITUDE and len(p) >= 6:
            pitch, roll, yaw = struct.unpack_from(">hhh", p, 0)  # rad * 10000
            t.attitude = tuple(math.degrees(v / 10000) for v in (pitch, roll, yaw))

        elif frame_type == FRAME_GPS and len(p) >= 15:
            lat, lon, speed, heading, alt, sats = struct.unpack_from(">iiHHHB", p, 0)
            t.gps = (lat / 1e7, lon / 1e7, speed / 10, heading / 100, alt - 1000, sats)

        elif frame_type == FRAME_LINK_STATS and len(p) >= 10:
            (rssi1, rssi2, lq, snr, ant, rf_mode, tx_pwr,
             d_rssi, d_lq, d_snr) = struct.unpack_from(">BBBbBBBBBb", p, 0)
            t.link = {
                "rssi": rssi_dbm(rssi1 if ant == 0 else rssi2),
                "lq": lq,
                "snr": snr,
                "down_rssi": rssi_dbm(d_rssi),
                "down_lq": d_lq,
            }

        elif frame_type == FRAME_FLIGHT_MODE:
            t.flight_mode = p.split(b"\x00", 1)[0].decode("ascii", "replace")

        elif frame_type == FRAME_RADIO_ID and len(p) >= 11 and p[2] == 0x10:
            # Extended frame: dest, origin, subtype, rate(int32), offset(int32), 0.1 us units
            rate, _offset = struct.unpack_from(">ii", p, 3)
            period = rate / 1e7
            if 0.0005 <= period <= 0.05:
                t.packet_period = max(period, 1.0 / MAX_RATE_HZ)

    except struct.error as e:
        print(f"[warn] malformed frame 0x{frame_type:02X}: {e}")


def read_telemetry(ser, buf, t):
    n = ser.in_waiting
    if n:
        buf.extend(ser.read(n))

    while len(buf) >= 2:
        if buf[0] not in VALID_START_BYTES:
            del buf[0]
            continue

        length = buf[1]
        if length < 2 or length > CRSF_MAX_LEN_FIELD:
            del buf[0]                     # Impossible length: not a real header
            continue

        total = length + 2
        if len(buf) < total:
            break                          # Wait for the rest of the frame

        body = bytes(buf[2:total - 1])     # type + payload
        if crc8_dvb_s2(body) != buf[total - 1]:
            del buf[0]                     # Bad CRC: resync on next byte
            continue

        del buf[:total]
        # Note: on a half-duplex bridge you may see echoes of your own RC
        # frames (type 0x16); handle_frame simply ignores them.
        handle_frame(body[0], body[1:], t)


def print_status(t):
    parts = []
    if t.flight_mode:
        parts.append(f"mode {t.flight_mode}")
    if t.battery:
        v, a, mah, pct = t.battery
        parts.append(f"bat {v:.1f}V {a:.1f}A {mah}mAh {pct}%")
    if t.link:
        parts.append(f"RSSI {t.link['rssi']}dBm LQ {t.link['lq']}%")
    if t.attitude:
        pitch, roll, yaw = t.attitude
        parts.append(f"P {pitch:+.1f}° R {roll:+.1f}° Y {yaw:+.1f}°")
    if t.gps:
        lat, lon, spd, hdg, alt, sats = t.gps
        parts.append(f"GPS {lat:.6f},{lon:.6f} {alt}m {spd:.1f}km/h {hdg:.0f}° {sats}sat")
    parts.append(f"tx {1 / t.packet_period:.0f}Hz")
    print(" | ".join(parts))


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    ser = serial.Serial(PORT, BAUD, timeout=0, write_timeout=0.05)
    rx = bytearray()
    t = Telemetry()

    # AETR + AUX. AUX1 (CH5) is the arm switch: keep it LOW at startup.
    # To arm: throttle low and channels[4] = 2000.
    channels = [1500, 1500, 1000, 1500] + [1000] * 12
    disarm = [1500, 1500, 1000, 1500] + [1000] * 12

    print("Streaming CRSF to ELRS TX module (Ctrl+C to stop)...")
    next_tick = time.perf_counter()
    next_print = next_tick

    try:
        while True:
            ser.write(rc_frame(channels))
            read_telemetry(ser, rx, t)

            now = time.perf_counter()
            if now >= next_print:
                print_status(t)
                next_print = now + STATUS_PRINT_S

            # Deadline-based scheduling: no cumulative drift
            next_tick += t.packet_period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()  # Fell behind: resync

    except KeyboardInterrupt:
        print("\nStopping: sending disarm frames...")
    finally:
        # ~500 ms of disarm
        for _ in range(max(1, int(0.5 / t.packet_period))):
            try:
                ser.write(rc_frame(disarm))
            except serial.SerialException:
                break
            time.sleep(t.packet_period)
        ser.close()


if __name__ == "__main__":
    main()