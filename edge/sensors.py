"""
Sensor hardware abstraction layer.

Covers the two non-camera sensors in the hardware list:

    MAX30100  - pulse oximeter, gives heart rate and (from beat-to-beat
                intervals) heart rate variability
    MPU6050   - accelerometer + gyroscope, gives head tilt / nodding posture

Three interchangeable hubs are provided so the same detection code runs
whether or not the hardware is plugged in:

    SerialSensorHub    - ESP32/Arduino node streaming JSON lines over USB or
                         UART. This is the normal deployment path and matches
                         firmware/esp32_sensor_node.
    I2CSensorHub       - sensors wired straight to a Raspberry Pi I2C bus.
    SimulatedSensorHub - physiologically plausible synthetic operator, used
                         for bench demos and the automated tests.

Whichever hub is active is named in every telemetry packet, so simulated
readings are always identifiable as such.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SensorSample:
    """One synchronised reading from the sensor hub."""

    heart_rate: Optional[float] = None       # bpm
    spo2: Optional[float] = None             # %
    hrv_rmssd: Optional[float] = None        # ms, root mean square of successive differences
    tilt_pitch: float = 0.0                  # degrees, forward nod
    tilt_roll: float = 0.0                   # degrees, sideways lean
    tilt_magnitude: float = 0.0              # combined tilt from vertical
    accel: tuple = (0.0, 0.0, 1.0)           # g
    gyro: tuple = (0.0, 0.0, 0.0)            # deg/s
    valid: bool = False
    source: str = "none"
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Heart-rate signal processing
# --------------------------------------------------------------------------


class HeartRateAnalyser:
    """
    Turns a stream of beat timestamps into heart rate and HRV.

    RMSSD (root mean square of successive RR-interval differences) is the
    standard short-window HRV metric. It falls as the sympathetic nervous
    system disengages, which is exactly the pattern seen when an operator is
    sliding toward sleep, so it complements the camera signals rather than
    duplicating them.
    """

    def __init__(self, window_beats: int = 30):
        self._rr_ms: deque = deque(maxlen=window_beats)
        self._last_beat: Optional[float] = None

    def add_beat(self, timestamp: float) -> None:
        if self._last_beat is not None:
            rr = (timestamp - self._last_beat) * 1000.0
            # Reject physiologically impossible intervals (motion artefacts).
            if 300.0 <= rr <= 2000.0:
                self._rr_ms.append(rr)
        self._last_beat = timestamp

    def heart_rate(self) -> Optional[float]:
        if len(self._rr_ms) < 3:
            return None
        return 60000.0 / float(np.mean(self._rr_ms))

    def rmssd(self) -> Optional[float]:
        if len(self._rr_ms) < 5:
            return None
        diffs = np.diff(np.asarray(self._rr_ms, dtype=float))
        return float(math.sqrt(float(np.mean(diffs ** 2))))

    def reset(self) -> None:
        self._rr_ms.clear()
        self._last_beat = None


def tilt_from_accel(ax: float, ay: float, az: float) -> tuple:
    """
    Convert a 3-axis accelerometer reading (in g) to pitch and roll.

    With the MPU6050 mounted on the operator headset / cap peak, gravity gives
    an absolute reference, so tilt does not drift the way a gyro-only estimate
    would.
    """
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    roll = math.degrees(math.atan2(ay, az if abs(az) > 1e-6 else 1e-6))
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180
    magnitude = math.degrees(math.acos(max(-1.0, min(1.0, az / max(
        1e-6, math.sqrt(ax * ax + ay * ay + az * az))))))
    return pitch, roll, magnitude


# --------------------------------------------------------------------------
# Simulated hub
# --------------------------------------------------------------------------


class SimulatedSensorHub:
    """
    Synthetic MAX30100 + MPU6050.

    Generates a resting heart rate with respiratory sinus arrhythmia, a slow
    downward drift as the shift wears on, and discrete nod-off events in which
    the head pitches forward and HR drops. Not a physiological model - a
    repeatable stand-in that exercises every branch of the fatigue logic.
    """

    name = "simulated"

    def __init__(self, baseline_hr: float = 74.0,
                 scenario: str = "progressive_fatigue", seed: int = 11):
        self.baseline_hr = baseline_hr
        self.scenario = scenario
        self._rng = np.random.default_rng(seed)
        self._t0 = time.time()
        self._analyser = HeartRateAnalyser()
        self._next_beat = time.time()
        self._nod_until = 0.0
        self._next_nod = 45.0
        self._coupling = 0.0        # see couple_to_observed_state()
        self._running = False

    def start(self) -> None:
        self._running = True

    def reset_clock(self, now: Optional[float] = None) -> None:
        """Restart the scripted shift; see SyntheticBackend.reset_clock."""
        now = time.time() if now is None else now
        self._t0 = now
        self._next_beat = now
        self._nod_until = 0.0
        self._next_nod = 45.0

    def stop(self) -> None:
        self._running = False

    def couple_to_observed_state(self, normalised_fatigue: float) -> None:
        """
        Let the camera drive the simulated vitals.

        When a real webcam is used with simulated sensors - the usual bench
        setup - this keeps the two signal families physically coherent: if the
        operator really is closing their eyes on camera, the fake heart rate
        sags with them instead of contradicting the video. Purely a demo aid;
        it does nothing when real sensors are attached.
        """
        self._coupling = float(np.clip(normalised_fatigue, 0.0, 1.0))

    def _fatigue_level(self, elapsed: float) -> float:
        if self.scenario == "alert":
            scripted = 0.05
        elif self.scenario == "microsleep_event":
            scripted = 0.15 if elapsed < 20 else 0.85
        else:
            scripted = float(np.clip(elapsed / 120.0, 0.0, 0.95))
        return max(scripted, self._coupling)

    def read(self, now: Optional[float] = None) -> SensorSample:
        now = time.time() if now is None else now
        elapsed = now - self._t0
        fatigue = self._fatigue_level(elapsed)

        # --- discrete nod-off events -------------------------------------
        if elapsed > self._next_nod and fatigue > 0.4:
            self._nod_until = now + 1.2 + 2.2 * fatigue
            self._next_nod = elapsed + max(12.0, 40.0 * (1.0 - fatigue) + 8.0)
        nodding = now < self._nod_until

        # --- heart rate: drifts down with fatigue, HRV collapses ----------
        target_hr = self.baseline_hr - 14.0 * fatigue
        if nodding:
            target_hr -= 6.0
        respiratory = 2.2 * math.sin(2 * math.pi * elapsed / 4.5)   # RSA
        instant_hr = target_hr + respiratory + float(self._rng.normal(0, 1.1))
        instant_hr = float(np.clip(instant_hr, 40.0, 130.0))

        # Emit beats at the instantaneous rate so HRV is computed the same way
        # it would be from a real MAX30100 beat interrupt.
        while self._next_beat <= now:
            jitter_ms = float(self._rng.normal(0, 26.0 * (1.0 - 0.75 * fatigue)))
            rr_s = max(0.35, 60.0 / instant_hr + jitter_ms / 1000.0)
            self._analyser.add_beat(self._next_beat)
            self._next_beat += rr_s

        # --- IMU ----------------------------------------------------------
        if nodding:
            pitch = 20.0 + 16.0 * fatigue + float(self._rng.normal(0, 2.0))
            roll = 6.0 * fatigue + float(self._rng.normal(0, 2.5))
        else:
            pitch = float(self._rng.normal(0, 3.2)) + 4.0 * fatigue
            roll = float(self._rng.normal(0, 2.6))

        pitch_r, roll_r = math.radians(pitch), math.radians(roll)
        ax = -math.sin(pitch_r)
        ay = math.sin(roll_r) * math.cos(pitch_r)
        az = math.cos(roll_r) * math.cos(pitch_r)
        _p, _r, magnitude = tilt_from_accel(ax, ay, az)

        hr = self._analyser.heart_rate()
        return SensorSample(
            heart_rate=hr if hr is not None else instant_hr,
            spo2=float(np.clip(97.5 - 1.5 * fatigue + self._rng.normal(0, 0.4), 90, 100)),
            hrv_rmssd=self._analyser.rmssd(),
            tilt_pitch=pitch, tilt_roll=roll, tilt_magnitude=magnitude,
            accel=(ax, ay, az),
            gyro=(float(self._rng.normal(0, 3)), float(self._rng.normal(0, 3)),
                  float(self._rng.normal(0, 2))),
            valid=True, source=self.name, timestamp=now,
        )


# --------------------------------------------------------------------------
# Serial hub (ESP32 / Arduino sensor node)
# --------------------------------------------------------------------------


class SerialSensorHub:
    """
    Reads newline-delimited JSON from the firmware in firmware/esp32_sensor_node.

    Expected line format:
        {"hr":72.4,"spo2":97.1,"beat":1,"ax":0.01,"ay":-0.03,"az":0.99,
         "gx":0.4,"gy":-1.1,"gz":0.2}

    A background thread drains the port so a slow or noisy link can never stall
    the main detection loop; read() always returns the most recent sample.
    """

    name = "serial"

    def __init__(self, port: str = "auto", baud: int = 115200, timeout: float = 1.0):
        import serial                     # pyserial, imported lazily
        from serial.tools import list_ports

        if port == "auto":
            candidates = [p.device for p in list_ports.comports()
                          if any(tag in (p.description or "").lower()
                                 for tag in ("ch340", "cp210", "usb serial",
                                             "silicon labs", "arduino", "esp32"))]
            if not candidates:
                raise RuntimeError("No ESP32/Arduino serial port found")
            port = candidates[0]

        self.port = port
        self._serial = serial.Serial(port, baud, timeout=timeout)
        self._analyser = HeartRateAnalyser()
        self._latest = SensorSample(source=self.name)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def available() -> bool:
        try:
            from serial.tools import list_ports
            return len(list(list_ports.comports())) > 0
        except Exception:
            return False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name="serial-sensor-hub")
        self._thread.start()

    def _pump(self) -> None:
        while self._running:
            try:
                raw = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if not raw or not raw.startswith("{"):
                    continue
                payload = json.loads(raw)
            except Exception:
                continue

            now = time.time()
            if payload.get("beat"):
                self._analyser.add_beat(now)

            ax = float(payload.get("ax", 0.0))
            ay = float(payload.get("ay", 0.0))
            az = float(payload.get("az", 1.0))
            pitch, roll, magnitude = tilt_from_accel(ax, ay, az)

            hr = payload.get("hr")
            derived_hr = self._analyser.heart_rate()
            sample = SensorSample(
                heart_rate=float(hr) if hr else derived_hr,
                spo2=float(payload["spo2"]) if payload.get("spo2") else None,
                hrv_rmssd=self._analyser.rmssd(),
                tilt_pitch=pitch, tilt_roll=roll, tilt_magnitude=magnitude,
                accel=(ax, ay, az),
                gyro=(float(payload.get("gx", 0.0)), float(payload.get("gy", 0.0)),
                      float(payload.get("gz", 0.0))),
                valid=True, source=self.name, timestamp=now,
            )
            with self._lock:
                self._latest = sample

    def read(self, now: Optional[float] = None) -> SensorSample:
        with self._lock:
            sample = self._latest
        # Treat a silent link as invalid rather than reporting stale vitals.
        if (time.time() if now is None else now) - sample.timestamp > 3.0:
            return SensorSample(source=self.name, valid=False)
        return sample

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
        try:
            self._serial.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Direct I2C hub (sensors wired to a Raspberry Pi)
# --------------------------------------------------------------------------


class I2CSensorHub:
    """
    MPU6050 read directly over the Pi I2C bus.

    The MAX30100 needs its own interrupt-driven driver; when it is not present
    the hub still supplies posture data and simply reports no heart rate, and
    the fatigue engine degrades gracefully by re-weighting the remaining
    indicators.
    """

    name = "i2c"
    _MPU_ADDR = 0x68
    _PWR_MGMT_1 = 0x6B
    _ACCEL_XOUT_H = 0x3B

    def __init__(self, bus_number: int = 1):
        import smbus2                      # imported lazily
        self._bus = smbus2.SMBus(bus_number)
        self._bus.write_byte_data(self._MPU_ADDR, self._PWR_MGMT_1, 0)   # wake
        time.sleep(0.05)
        self._analyser = HeartRateAnalyser()

    @staticmethod
    def available() -> bool:
        try:
            import smbus2
            bus = smbus2.SMBus(1)
            bus.read_byte_data(I2CSensorHub._MPU_ADDR, 0x75)   # WHO_AM_I
            bus.close()
            return True
        except Exception:
            return False

    def start(self) -> None:
        pass

    def _read_word(self, reg: int) -> int:
        high = self._bus.read_byte_data(self._MPU_ADDR, reg)
        low = self._bus.read_byte_data(self._MPU_ADDR, reg + 1)
        value = (high << 8) | low
        return value - 65536 if value >= 0x8000 else value

    def read(self, now: Optional[float] = None) -> SensorSample:
        try:
            ax = self._read_word(self._ACCEL_XOUT_H) / 16384.0
            ay = self._read_word(self._ACCEL_XOUT_H + 2) / 16384.0
            az = self._read_word(self._ACCEL_XOUT_H + 4) / 16384.0
            gx = self._read_word(0x43) / 131.0
            gy = self._read_word(0x45) / 131.0
            gz = self._read_word(0x47) / 131.0
        except Exception:
            return SensorSample(source=self.name, valid=False)

        pitch, roll, magnitude = tilt_from_accel(ax, ay, az)
        return SensorSample(
            heart_rate=self._analyser.heart_rate(),
            hrv_rmssd=self._analyser.rmssd(),
            tilt_pitch=pitch, tilt_roll=roll, tilt_magnitude=magnitude,
            accel=(ax, ay, az), gyro=(gx, gy, gz),
            valid=True, source=self.name,
        )

    def stop(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def create_sensor_hub(cfg, logger=None):
    """
    Build the sensor hub named in the config.

    'auto' tries the real hardware first (serial node, then direct I2C) and
    falls back to simulation, printing exactly which path was taken so a demo
    never silently pretends to have hardware it does not have.
    """
    driver = str(cfg.sensors.get("driver", "auto")).lower()
    sim_cfg = cfg.sensors.get("simulation", {})

    def _log(message: str) -> None:
        if logger:
            logger(message)

    def _simulated():
        return SimulatedSensorHub(
            baseline_hr=float(sim_cfg.get("baseline_hr", 74.0)),
            scenario=str(sim_cfg.get("scenario", "progressive_fatigue")))

    if driver == "simulated":
        return _simulated()

    if driver in ("auto", "serial"):
        try:
            hub = SerialSensorHub(port=str(cfg.sensors.serial.get("port", "auto")),
                                  baud=int(cfg.sensors.serial.get("baud", 115200)))
            _log("sensor hub: ESP32 node on %s" % hub.port)
            return hub
        except Exception as exc:
            if driver == "serial":
                raise
            _log("sensor hub: no serial node (%s)" % exc)

    if driver in ("auto", "i2c"):
        try:
            hub = I2CSensorHub()
            _log("sensor hub: MPU6050 on I2C bus 1")
            return hub
        except Exception as exc:
            if driver == "i2c":
                raise
            _log("sensor hub: no I2C sensors (%s)" % exc)

    _log("sensor hub: SIMULATED (no hardware detected)")
    return _simulated()
