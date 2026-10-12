#!/usr/bin/env python3
"""
CRSF link diagnostic. Sends neutral, DISARMED RC frames and reports once per
second which CRSF frames come back, who sent them, and the link statistics.

Put this file next to crsf_pc_control.py and stop that script before running
this one (only one program can open the serial port).
"""

import time
from collections import Counter

import serial

from crsf_pc_control import (
    BAUD, CRSF_MAX_LEN_FIELD, PORT, VALID_START_BYTES, crc8_dvb_s2, rc_frame, rssi_dbm
)

FRAME_NAMES = {
    0x08: "battery (from FC)",
    0x14: "link stats (from TX module)",
    0x16: "RC channels (echo of our own)",
    0x1E: "attitude (from FC)",
    0x02: "GPS (from FC)",
    0x21: "flight mode (from FC)",
    0x29: "device info",
    0x3A: "timing sync (from TX module)",
}

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0)
    buf = bytearray()
    counts = Counter()
    last = {}
    rx_bytes = junk = bad_crc = 0

    channels = [1500, 1500, 1000, 1500] + [1000] * 12   # AUX1 low = disarmed
    next_report = time.perf_counter() + 1.0
    print(f"Diagnosing {PORT} @ {BAUD} (Ctrl+C to stop)...")

    try:
        while True:
            ser.write(rc_frame(channels))

            n = ser.in_waiting
            if n:
                data = ser.read(n)
                rx_bytes += len(data)
                buf.extend(data)

            while len(buf) >= 2:
                if buf[0] not in VALID_START_BYTES:
                    del buf[0]
                    junk += 1
                    continue
                length = buf[1]
                if length < 2 or length > CRSF_MAX_LEN_FIELD:
                    del buf[0]
                    junk += 1
                    continue
                total = length + 2
                if len(buf) < total:
                    break
                body = bytes(buf[2:total - 1])
                if crc8_dvb_s2(body) != buf[total - 1]:
                    del buf[0]
                    bad_crc += 1
                    continue
                counts[(buf[0], body[0])] += 1
                last[body[0]] = body[1:]
                del buf[:total]

            now = time.perf_counter()
            if now >= next_report:
                print(f"\n--- last 1 s: {rx_bytes} bytes in, "
                      f"{junk} junk bytes, {bad_crc} bad CRC ---")
                if not counts:
                    print("  no valid CRSF frames received")
                for (addr, ftype), c in sorted(counts.items()):
                    name = FRAME_NAMES.get(ftype, "?")
                    print(f"  addr 0x{addr:02X}  type 0x{ftype:02X}  {name:32s} x{c}")

                p = last.get(0x14)
                if p is not None and len(p) >= 10:
                    snr = int.from_bytes(p[3:4], "big", signed=True)
                    print(f"  LINK: uplink RSSI {rssi_dbm(p[0])}/ {rssi_dbm(p[1])} dBm, uplink LQ {p[2]}%, "
                          f"SNR {snr} dB, RF mode {p[5]}, TX power idx {p[6]}, "
                          f"downlink RSSI {rssi_dbm(p[7])} dBm, downlink LQ {p[8]}%")

                fm = last.get(0x21)
                if fm is not None:
                    mode = fm.split(b"\x00", 1)[0].decode("ascii", "replace")
                    print(f"  FC flight mode: {mode}")

                counts.clear()
                last.clear()
                rx_bytes = junk = bad_crc = 0
                next_report = now + 1.0

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()