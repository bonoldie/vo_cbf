# CRSF Drone control

## Used channels

> ON/OFF are defined as >/<1700

| Mode       | Input | Activation | Note                                |
| ---------- | ----- | ---------- | ----------------------------------- |
| Arm        | AUX 1 | ON/OFF     |                                     |
| Angle      | AUX 2 | < 1300     |                                     |
| Horizon    | AUX 2 | > 1700     |                                     |
| Mag        | AUX 3 | ON/OFF     | Heading lok to magnetometer heading |
| Failsafe   | AUX 4 | ON/OFF     | Enter stage 2 failsafe manually     |
| AltHold    | AUX 5 | < 1300     |                                     |
| PosHold    | AUX 5 | > 1700     |                                     |
| GPSRescue  | AUX 6 | ON/OFF     |                                     |
| AutoPilot  | AUX 7 | ON/OFF     | Check out the FlightPlan            |
| Beeper     | AUX 8 | ON/OFF     |                                     |
| VTXPitMode | AUX 9 | ON/OFF     |                                     |

