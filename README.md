# Real-Time Monitoring and Alerting System for Crane Operator Fatigue

**Embedded Systems — Digital Assignment 1**

| | |
|---|---|
| **Faizan Shahid** | 24BCE1900 |
| **Kritagya Vij** | 24BCE5083 |
| **Ekansh Goyal** | 24BCE1943 |

A working implementation of the system described in the project abstract: a
cabin-mounted edge unit that watches a crane operator for signs of fatigue,
alerts them locally within the cabin, and escalates to a control-room dashboard
where a supervisor can act.

Everything in the abstract's operational flow is implemented and running —
calibration, EAR/MAR extraction, head-pose and heart-rate monitoring, composite
fatigue scoring, three-level classification, local annunciation, remote
alerting, database logging, supervisor action, and the post-shift report.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python run_demo.py
```

That single command starts the control-room server, seeds eight hours of shift
history so the report page has something to show, opens the dashboard in your
browser, and launches the cabin unit. **It works with no hardware at all** — if
there is no webcam it runs a scripted operator; if there are no sensors it
simulates them, and says so on screen rather than pretending otherwise.

If you do have a webcam, it is used automatically. Sit in front of it and close
your eyes for a second or two to drive the score up.

| What | Where |
|---|---|
| Live supervisor dashboard | <http://127.0.0.1:5000/> |
| Post-shift report | <http://127.0.0.1:5000/report> |
| Cabin HUD | opens as its own window |

To run the 65 automated tests:

```bash
python -m unittest discover tests -v
```

---

## What the system does

```
       CABIN (edge unit)                          CONTROL ROOM
  ┌───────────────────────────┐              ┌──────────────────────┐
  │ camera ─► face landmarks  │              │  Flask server        │
  │            │              │   Wi-Fi/GSM  │    │                 │
  │            ▼              │   telemetry  │    ▼                 │
  │  EAR · MAR · head pose    │─────────────►│  SQLite log          │
  │            │              │              │    │                 │
  │ MAX30100 ─►│◄─ MPU6050    │              │    ├─► live dashboard│
  │            ▼              │              │    └─► shift report  │
  │   composite fatigue score │              │                      │
  │            │              │◄─────────────│  supervisor commands │
  │            ▼              │   commands   │  (ack / rotate /halt)│
  │  Normal · Warning · Critical             └──────────────────────┘
  │            │              │
  │            ▼              │
  │  LED · buzzer · vibration │
  └───────────────────────────┘
```

**Four independent fatigue indicators** are combined, so no single sensor
failure blinds the system:

| Indicator | Measured from | Detects |
|---|---|---|
| Drowsiness | Eye Aspect Ratio + PERCLOS | microsleeps, slow eyelid droop |
| Yawning | Mouth Aspect Ratio | early-stage fatigue |
| Posture | MPU6050 tilt (camera as backup) | nodding off |
| Vitals | MAX30100 heart rate + HRV | autonomic disengagement |

---

## Running the pieces separately

```bash
python -m server.app
```
Control room only — dashboard, report, ingest API, command downlink.

```bash
python -m edge.node
```
Cabin unit only. Useful flags:

| Flag | Effect |
|---|---|
| `--backend mediapipe\|haar\|synthetic` | force a landmark backend |
| `--camera 1` or `--camera clip.mp4` | pick a camera, or replay a video file |
| `--sensors serial\|i2c\|simulated` | force a sensor hub |
| `--scenario microsleep_event` | scripted operator behaviour |
| `--headless` | no GUI window (for a Raspberry Pi over SSH) |
| `--no-telemetry` | run standalone, never contact the server |
| `--set scoring.thresholds.high=50` | override any config value |

In the cabin HUD window: `q` quit · `c` recalibrate · `r` reset counters ·
`h` simulate a link failure (demonstrates store-and-forward buffering).

```bash
python -m tools.simulate_shift --reset --hours 8 --cranes 3
```
Regenerate demo history for the report page.

---

## Repository layout

```
config/system_config.yaml   every tunable constant, with the reasoning
edge/                       the cabin unit
  node.py                     main real-time loop (algorithm steps 1-16)
  vision.py                   face landmarks, EAR, MAR, head pose
  sensors.py                  MAX30100 + MPU6050 abstraction layer
  fatigue.py                  scoring engine and state machine
  alerts.py                   LED / buzzer / vibration driver
  telemetry.py                store-and-forward uplink, command downlink
  hud.py                      the in-cabin heads-up display
server/                     the control room
  app.py                      Flask API + SSE push
  database.py                 SQLite schema and queries
  templates/, static/         dashboard and shift report (no CDN needed)
firmware/esp32_sensor_node/ Arduino sketch for the sensor board
tools/simulate_shift.py     demo history generator
tests/                      65 automated tests
docs/algorithm.md           the detection logic, and where it departs from
                            the abstract — read this one
docs/hardware.md            wiring, bill of materials, deployment notes
```

---

## Designed to degrade, not to fail

A safety system that stops working when a part is missing is worse than no
system, because it is trusted. Every layer has a documented fallback and
**always reports which one is active** — the dashboard says
*"simulated input, not a live measurement"* whenever that is the case.

| Missing | What happens |
|---|---|
| MediaPipe not installed | falls back to OpenCV Haar cascades |
| No camera | scripted operator, frames watermarked `SIMULATED` |
| No ESP32 / no I2C sensors | simulated vitals, labelled as such |
| No Raspberry Pi GPIO | alerts go to the console and the PC sound card |
| Network link drops | telemetry buffers to disk, replays on reconnect |
| Heart-rate sensor fails | dropped from the weighted mean, not scored zero |
| Bad calibration baseline | refused, falls back to absolute thresholds, says so |

That last row matters: scoring a missing signal as zero would make a fatigued
operator appear *less* tired the moment a sensor failed. There is a test for it
(`test_losing_the_heart_rate_sensor_does_not_deflate_the_score`).

---

## Honest limitations

Stated plainly, because a review should not have to discover them.

- **The fatigue score is not clinically validated.** The weights and thresholds
  in `config/system_config.yaml` are reasoned engineering defaults built on
  published EAR/PERCLOS work; validating them needs a labelled dataset of real
  operators, which is Phase 7 of the project timeline.
- **No ML model is trained.** The abstract lists TensorFlow Lite. What is
  implemented is a deterministic weighted-indicator model, which is
  interpretable, tunable and testable without training data. The scoring stage
  is isolated in `FatigueEngine`, so a trained classifier can replace it
  without touching the rest of the system. This is a deliberate choice, not an
  omission — an untrained model would have been a worse deliverable than a
  transparent one.
- **The halt command is advisory.** It raises the cabin alarm and is logged,
  but does not interlock the crane drive. Interlocking a lifting appliance is a
  functional-safety change requiring a rated safety controller and a site
  sign-off, not a Python process.
- **The Haar fallback is noticeably less accurate** than MediaPipe. It exists
  so the pipeline still runs on a bare OpenCV install, not as an equal.
- **Simulated data is simulated.** `tools/simulate_shift.py` generates
  physiologically plausible inputs, but the states and alerts it produces come
  from the same `FatigueEngine` that runs live — the driver is fake, the
  detector is not.
- **Privacy.** The camera watches a worker for a whole shift. No video is ever
  stored or transmitted — only derived scalars (EAR, MAR, tilt, heart rate) —
  but any real deployment needs worker consent and a data-retention policy.

---

## Where the code departs from the abstract

Four places, each because the pseudocode as written would misbehave in
practice. All four are argued in full in [`docs/algorithm.md`](docs/algorithm.md):

1. **Sliding evaluation window** instead of resetting counters at each window
   boundary — a tumbling reset silences a live alert the instant the window
   rolls over.
2. **Duration gates and refractory periods** on every threshold — without them
   one yawn is counted once per frame, roughly 45 times.
3. **Dominant-indicator escalation** on top of the weighted sum — a plain
   weighted mean can never let one indicator raise the alarm, so an operator
   with their eyes shut 40 % of the minute would cap at 45/100 and never be
   called Critical.
4. **Hysteresis and dwell** in the classifier — otherwise the state chatters
   between Warning and Normal while the score sits on a threshold.
