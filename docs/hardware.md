# Hardware build and deployment

Everything the software needs to talk to real hardware is already implemented
and selected automatically at startup (`sensors.driver: auto`,
`alerts.driver: auto`). This document covers what to buy, how to wire it, and
what to expect when the system leaves the bench.

---

## 1. Bill of materials

| # | Component | Purpose | Approx. cost (INR) |
|---|---|---|---|
| 1 | Raspberry Pi 4 (4 GB) | Edge processing unit | 4,500 |
| 1 | Pi Camera Module 3 **or** any USB webcam | Operator face capture | 1,800 |
| 1 | IR LED illuminator ring, 850 nm | Night / low-light operation | 350 |
| 1 | ESP32 DevKit v1 | Sensor sampling + annunciator driver | 450 |
| 1 | MAX30100 breakout | Heart rate + SpO₂ | 300 |
| 1 | MPU6050 breakout | Head tilt / nodding | 150 |
| 1 | Piezo buzzer, 12 mm | Audible alert | 30 |
| 1 | Coin vibration motor, 3 V | Haptic alert (noisy cabins) | 90 |
| 3 | LEDs, 5 mm (green / yellow / red) | Status indication | 15 |
| 3 | Resistors, 220 Ω | LED current limiting | 5 |
| 1 | 2N2222 transistor + 1N4148 diode | Vibration motor driver | 20 |
| 1 | Buck converter, 24 V → 5 V, 3 A | Power from the crane supply | 300 |
| 1 | Electrolytic cap, 1000 µF 35 V | Input surge smoothing | 40 |
| 1 | Articulated mounting bracket | Fixes the camera facing the operator | 400 |
| 1 | ABS enclosure, IP54 | Dust protection | 500 |
| | | **Total** | **≈ 8,950** |

A Jetson Nano can replace the Pi if a trained CNN is added later; nothing in the
software changes.

---

## 2. Wiring

### ESP32 sensor node

```
                    ESP32 DevKit v1
                   ┌───────────────┐
   MAX30100 SDA ───┤ GPIO21   3V3  ├─── MAX30100 VIN, MPU6050 VCC
   MPU6050  SDA ───┤               │
                   │               │
   MAX30100 SCL ───┤ GPIO22   GND  ├─── common ground
   MPU6050  SCL ───┤               │
                   │               │
   green  LED  ────┤ GPIO25        │   (each through 220 Ω)
   yellow LED  ────┤ GPIO26        │
   red    LED  ────┤ GPIO27        │
                   │               │
   buzzer      ────┤ GPIO23        │
   vibration   ────┤ GPIO19        │   (through 2N2222 + flyback diode)
                   │               │
   USB ────────────┤ micro-USB     │──► Raspberry Pi (data + 5 V)
                   └───────────────┘
```

Both sensors share the I²C bus — the MAX30100 is at `0x57` and the MPU6050 at
`0x68`, so there is no address clash. Fit 4.7 kΩ pull-ups on SDA and SCL if the
breakout boards do not already carry them.

**The vibration motor must not be driven directly from a GPIO pin.** It is an
inductive load that will exceed the pin current limit and induce a back-EMF
spike on switch-off. Use the transistor, and fit the flyback diode across the
motor.

### Alternative: sensors direct to the Pi

`I2CSensorHub` supports the MPU6050 on the Pi's own I²C bus (GPIO2 = SDA,
GPIO3 = SCL) and `GpioAlertDriver` drives the LEDs and buzzer from the pins in
`config/system_config.yaml`. This drops the ESP32 and its cost.

It is the **less good** option, and the code prefers the serial node for a
reason: the MAX30100 needs its beat interrupt serviced with tight timing, and a
Pi running OpenCV at 15 fps under a non-realtime Linux scheduler cannot promise
that. Beat detection degrades, and HRV — which depends on millisecond interval
accuracy — degrades faster. Put the sampling on the microcontroller.

---

## 3. Flashing the firmware

1. Arduino IDE → Boards Manager → install **esp32** by Espressif.
2. Library Manager → install:
   - `MAX30100lib` (OXullo Intersecans)
   - `Adafruit MPU6050` + `Adafruit Unified Sensor`
   - `ArduinoJson` (Benoit Blanchon)
3. Open `firmware/esp32_sensor_node/esp32_sensor_node.ino`.
4. Select **ESP32 Dev Module**, pick the port, upload.
5. Open the Serial Monitor at 115200 baud. You should see a boot line
   reporting which sensors initialised:

   ```json
   {"boot":1,"mpu6050":true,"max30100":true}
   ```

   followed by ~20 telemetry lines per second.

If `max30100` reports `false`, check the 3V3 supply — the breakout browns out
on 5 V logic-level supplies from some clone boards.

### Connecting it to the edge unit

Close the Serial Monitor first (it holds the port), then:

```bash
python -m edge.node --sensors serial
```

The node scans for an ESP32/Arduino by USB descriptor. To pin a specific port:

```bash
python -m edge.node --set sensors.serial.port=COM5      # Windows
python -m edge.node --set sensors.serial.port=/dev/ttyUSB0
```

Confirm it took: the startup log prints `sensor hub: ESP32 node on COM5`, and
the dashboard shows `Sensor hub: serial` rather than `simulated`.

---

## 4. Mounting in the cabin

- **Camera position.** Roughly 60–80 cm from the operator, slightly below eye
  level, angled up. A camera above eye level sees drooping eyelids as closed
  eyes and produces constant false positives.
- **Avoid backlight.** A camera pointed at the cabin glazing will silhouette
  the operator and the landmark detector will lose the face for most of the
  shift. Point it inward.
- **IR illumination** is required for night shifts. The 850 nm ring is
  invisible to the operator and does not affect their night vision. If the
  webcam has an IR-cut filter it must be removed for this to help.
- **IMU placement.** On the hard-hat brim or a headset band, not on the seat —
  a seat-mounted IMU measures the crane, not the operator.
- **Vibration.** Cranes vibrate. The MPU6050 filter bandwidth is set to 21 Hz
  in the firmware, well above head-nod frequencies and well below the
  structural vibration that would otherwise dominate.

---

## 5. Power

Crane cabin supplies are electrically hostile: 24 V nominal, with switching
transients from the hoist and slew drives.

- Use an **isolated** buck converter, not a linear regulator.
- Fit the 1000 µF capacitor across the 24 V input, close to the converter.
- Add a resettable fuse (1 A polyfuse) on the input.
- Give the Pi a clean shutdown path. Yanking power at the end of a shift will
  eventually corrupt the SD card. Either fit a UPS HAT, or mount the root
  filesystem read-only and keep only `data/` writable.

Measured draw of the assembled unit: ~4.5 W idle, ~7 W with the IR ring on.

---

## 6. Running headless on the Pi

```bash
python -m edge.node --headless --server http://192.168.1.50:5000
```

To start it automatically, `/etc/systemd/system/crane-fatigue.service`:

```ini
[Unit]
Description=Crane operator fatigue monitor (cabin unit)
After=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/crane-fatigue
ExecStart=/usr/bin/python3 -m edge.node --headless
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now crane-fatigue
journalctl -u crane-fatigue -f
```

`Restart=always` matters: a monitoring unit that has silently died is worse
than none, because the control room keeps showing its last known state. The
server guards against this too — a unit that stops reporting for 10 seconds is
marked **Offline** on the dashboard and its stale score is blanked rather than
left on screen looking live.

---

## 7. Network

The edge node buffers telemetry to disk whenever the link is down and replays
it on reconnect, so a moving crane passing through steel structure loses
nothing. Press `h` in the HUD window to simulate a link failure and watch the
buffer count rise, then drain when the link returns.

For a GSM deployment (SIM800L or a 4G dongle), only the server URL changes.
Bandwidth is modest — one JSON record per second, around 400 bytes, so roughly
**1.4 MB per hour per cabin**. Raise `telemetry.publish_interval_s` to reduce
it; state changes and alerts bypass the rate limiter and are always sent
immediately.

---

## 8. Commissioning checklist

Run through this once per cabin before trusting the system.

1. `python -m edge.node --sensors serial` starts and logs
   `sensor hub: ESP32 node on ...` — not `SIMULATED`.
2. The startup lamp test lights all three LEDs.
3. The dashboard shows the unit **online** with `Sensor hub: serial` and a
   real vision backend — no *"simulated input"* notice.
4. Calibration completes with the operator seated normally, and the logged
   baseline EAR is plausible (0.25–0.40).
5. Close your eyes for two seconds → `microsleep` appears in the alert feed and
   the buzzer sounds.
6. Yawn → `yawn` appears in the feed.
7. Cover the camera for three seconds → `Operator not visible` escalates.
8. Issue **Halt crane** from the dashboard → the cabin unit logs the command
   within about two seconds and the red LED holds.
9. Unplug the network → the buffer count rises. Reconnect → it drains and the
   history backfills with no gap.
10. Check the shift report shows the session, then have the supervisor
    acknowledge the outstanding alerts.
