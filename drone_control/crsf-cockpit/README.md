# CRSF Cockpit

Python 3 / GTK 3 desktop app for the supplied PC → serial bridge → ELRS TX → receiver → flight-controller setup. No web browser, no automatic port opening, and no automatic arming. Click Connect after choosing a port, or use Demo to explore without hardware.

## Install and run (Ubuntu / Debian)

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 python3-serial
python3 app.py --demo
python3 app.py --port /dev/ttyACM0 --baud 115200
```

Use the system Python that has the GTK packages installed. A virtual environment needs `--system-site-packages` to see them. Fedora equivalents: `python3-gobject gtk3 python3-cairo python3-pyserial`; Arch equivalents: `python-gobject gtk3 python-cairo python-pyserial`. Package availability may vary by distribution.

Only one program can own the serial port. Stop the supplied diagnostic/controller scripts first. If access is denied, grant your normal account access to the serial device using your distribution's serial-device group (often `dialout`, requiring a new login). Do not run the GUI as root.

The default 115200 baud is for your working bridge, not a claim about the TX-module UART baud. The bridge must already implement the module-side inversion, half duplex and turnaround. This app does not reconfigure FTDI, STM32 or ESP32 hardware.

## Controls and altitude behaviour

Drag the circular joystick: horizontal is roll, vertical is pitch. Release it to centre. Commands are body-relative tilt inputs, not Cartesian velocity or position setpoints. The initial limit is ±150 µs from 1500; adjust the authority slider as needed. The actual angle depends on the FC Angle-mode configuration, rates and channel interpretation. Verify pitch direction and use Invert pitch if necessary.

Yaw has a separate slider; releasing the mouse centres it. Throttle is persistent and starts at 1000 µs. Escape and window focus loss centre roll, pitch and yaw, **retaining throttle and ARM**. Centring the joystick levels an Angle-mode craft; it does not brake it to a fixed XY position.

For the requested planar steering, select Angle on AUX2 and AltHold on AUX5. The flight controller must actually support/configure AltHold and have suitable sensors. This app only requests the mode: it does not close an altitude loop, select a height in metres, or verify that the mode engaged. The explicit 1500-throttle button is for FC configurations where that is the neutral hold/climb input. Confirm your firmware's throttle semantics first; switching AltHold does not automatically change throttle or launch the aircraft. Takeoff and landing remain manual.

Mag requests your configured heading-lock mode. PosHold, GPSRescue and AutoPilot merely set their AUX inputs. There is no flight-plan uploader or Cartesian autopilot in this app. In particular, AutoPilot uses the FlightPlan already configured on your FC. The Mode telemetry card displays the reported text; there is no independent armed-state confirmation in these decoded frames.

## Exact channel map

ON=2000 µs; OFF=1000 µs. Midpoint=1500 µs. The supplied thresholds (>1700, <1300) are met by the endpoints.

| Channel | Function | Values / startup |
|---|---|---|
| CH1 | Roll | 1500 centre |
| CH2 | Pitch | 1500 centre |
| CH3 | Throttle | 1000 startup |
| CH4 | Yaw | 1500 centre |
| CH5 / AUX1 | Arm | OFF startup |
| CH6 / AUX2 | Angle / neither / Horizon | 1000 / 1500 / 2000; Angle startup |
| CH7 / AUX3 | Mag | OFF startup |
| CH8 / AUX4 | Stage-2 failsafe | OFF startup |
| CH9 / AUX5 | AltHold / neither / PosHold | 1000 / 1500 / 2000; neither startup |
| CH10 / AUX6 | GPSRescue | OFF startup |
| CH11 / AUX7 | AutoPilot | OFF startup |
| CH12 / AUX8 | Beeper | OFF startup |
| CH13 / AUX9 | VTX PitMode | OFF startup |
| CH14–16 / AUX10–12 | Unused | 1000 |

The on-screen strip shows requested channels, not a receiver readback. A latched watchdog overrides steering and AUX4 in the serial worker; the banner identifies this override. Check that your ELRS switch/channel mode actually transports all the used AUX channels at sufficient resolution and cadence.

## Telemetry and connection status

- Battery voltage/current, consumed capacity and remaining percentage.
- Pitch, roll, yaw in degrees.
- GPS coordinates, GPS altitude, ground speed, course and satellite count.
- Barometric altitude relative to calibration/start and vertical speed (if sent), plus variometer frames.
- Uplink/downlink LQ, RSSI, SNR, RF profile and reported TX power.
- Flight-mode text and per-topic age; values older than 3 seconds are labelled STALE.
- Separate serial-open, fresh positive radio-link LQ, and fresh FC-telemetry indications.
- Raw frame counts grouped by destination address/type, discarded bytes and CRC errors. Broadcast destination addresses do not identify the originating device. These common sensor frames are treated as evidence of FC telemetry in your topology, not authenticated FC identity.

RC echoes and timing sync never qualify as FC telemetry. The default RSSI compatibility mode preserves your signed-int8 ELRS handling for bytes ≥128. Use `--rssi-encoding magnitude` for the spec's unsigned negative-magnitude interpretation. Timing frames adjust the send period; offset is decoded but not used for phase alignment. Scheduling is best effort, not hard real time or full CRSFShot phase synchronization. Default 100 Hz, maximum 250 Hz, bounded to 70% of nominal serial bandwidth.

## ARM, failsafe and disconnect behaviour

Bench-test with propellers removed first. Verify every channel in the FC Receiver/Modes tabs, direction of movement, the bridge's loss-of-host behaviour, and the receiver/FC failsafe action.

ARM requires the setup checkbox, an open connection, fresh link telemetry with positive LQ, fresh FC telemetry, throttle ≤1050 µs, centred steering, and failsafe/rescue/autopilot off. This is a local guard, not a complete preflight check. A high AUX1 means ARM requested, not confirmed.

The serial worker runs independently of GTK. If the GUI heartbeat stops for >0.5 s while ARM is requested, it centres steering, retains throttle/ARM and sets AUX4 high, requesting your configured stage-2 failsafe. It latches until an explicit disarm; resumed GUI activity cannot silently clear it. A stale telemetry downlink alone is displayed as stale and does not automatically cut motors or trigger a failsafe.

If the process dies or serial writes fail, the app cannot deliver an AUX failsafe request. Your bridge must stop replaying old controls when its host disappears, and your receiver/FC must handle the resulting link loss. Test this chain before flight. The manual Stage-2 FAILSAFE button sets AUX4 high; its actual outcome depends on FC configuration. DISARM lowers AUX1 and throttle immediately and can stop motors in flight.

Disconnect/window close is refused while ARM is requested. Disarm first, confirm motors stop, then Disconnect. The app streams disarmed commands for approximately 300 ms before closing the port. Close the window after disconnect completes. No automatic reconnect or re-arm is performed. Demo uses synthetic telemetry and never opens a serial port.

## Files and verification

- `app.py`: GTK UI and input mapping.
- `crsf.py`: pure Python framing, parser and telemetry decoder.
- `link.py`: serial scheduler and watchdog.
- `test_crsf.py`: protocol and fake-serial integration tests.

```bash
python3 -m unittest -v
```

Nine tests passed during delivery: independent 11-bit unpacking, CRC known vector, fragmented/noisy stream resynchronization, partial-frame timeout, telemetry scales, timing frame validation, startup mappings, watchdog latch and serial-open failure. Python compilation passed. GTK was unavailable in the build environment, so the window was not runtime/visually tested. No real bridge, transmitter, FC or flight test was performed.

Protocol reference reviewed: https://raw.githubusercontent.com/tbs-fpv/tbs-crsf-spec/refs/heads/main/crsf.md (retrieved 2026-10-12).
