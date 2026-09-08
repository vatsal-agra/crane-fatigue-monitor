"""
Environment check - run this first on a new machine.

    python -m tools.check_setup

Reports what this laptop can actually do: which packages are installed, which
detection backend will be selected, whether the camera opens and how long it
takes, and whether the ports the demo needs are free.

The point is to find problems now rather than in front of an audience. Nothing
here is fatal on its own - the system has a fallback for every layer - so the
output is graded:

    [ ok ]    working
    [ note ]  running on a fallback; the demo still works, quality is lower
    [FAIL]    the demo will not run until this is fixed
"""
from __future__ import annotations

import importlib
import os
import platform
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, NOTE, FAIL = "[ ok ]", "[note]", "[FAIL]"
_results = {"ok": 0, "note": 0, "fail": 0}


def report(level: str, title: str, detail: str = "") -> None:
    _results["ok" if level is OK else "note" if level is NOTE else "fail"] += 1
    print("  %s %-34s %s" % (level, title, detail))


def section(name: str) -> None:
    print("\n" + name)
    print("  " + "-" * 66)


def check_python() -> None:
    section("Python")
    version = sys.version_info
    text = "%d.%d.%d  (%s)" % (version.major, version.minor, version.micro,
                               platform.system())
    if version < (3, 9):
        report(FAIL, "interpreter", text + "  - needs 3.9 or newer")
    elif version >= (3, 13):
        report(NOTE, "interpreter", text + "  - mediapipe has no 3.13 wheel yet")
    else:
        report(OK, "interpreter", text)


def check_packages() -> None:
    section("Packages")
    required = [("cv2", "opencv-python"), ("numpy", "numpy"),
                ("flask", "Flask"), ("requests", "requests"), ("yaml", "PyYAML")]
    for module, package in required:
        try:
            mod = importlib.import_module(module)
            report(OK, package, getattr(mod, "__version__", ""))
        except Exception:
            report(FAIL, package, "missing - run: pip install -r requirements.txt")

    optional = [("mediapipe", "mediapipe", "accurate landmarks"),
                ("serial", "pyserial", "ESP32 sensor node"),
                ("smbus2", "smbus2", "sensors on a Pi I2C bus"),
                ("paho.mqtt", "paho-mqtt", "MQTT telemetry"),
                ("RPi.GPIO", "RPi.GPIO", "real LEDs and buzzer")]
    for module, package, purpose in optional:
        try:
            importlib.import_module(module)
            report(OK, package, purpose)
        except Exception:
            report(NOTE, package, "not installed - %s unavailable" % purpose)


def check_backend() -> None:
    section("Detection backend")
    try:
        from edge.vision import (HaarCascadeBackend, MediaPipeBackend,
                                 create_backend)
    except Exception as exc:
        report(FAIL, "import edge.vision", str(exc)[:60])
        return

    if MediaPipeBackend.available():
        report(OK, "mediapipe FaceMesh", "478 landmarks - full accuracy")
    elif HaarCascadeBackend.available():
        report(NOTE, "OpenCV Haar cascades",
               "fallback - the pipeline runs, accuracy is lower")
    else:
        report(NOTE, "no landmark backend", "scripted operator only")

    try:
        backend = create_backend("auto")
        report(OK, "auto-selected backend", backend.name)
        backend.close()
    except Exception as exc:
        report(FAIL, "backend selection", str(exc)[:60])


def check_camera() -> None:
    section("Camera")
    try:
        import cv2

        from edge.vision import VideoSource
    except Exception as exc:
        report(FAIL, "opencv", str(exc)[:60])
        return

    started = time.time()
    try:
        source = VideoSource(0, 640, 480, True, allow_synthetic=True)
        elapsed = time.time() - started
        if source.is_synthetic:
            report(NOTE, "webcam", "none found - the demo uses a scripted operator")
        else:
            ok, frame = source.read()
            detail = "%s, opened in %.1fs" % (source.description, elapsed)
            if not ok:
                report(NOTE, "webcam", "opened but returned no frame")
            elif elapsed > 15:
                report(NOTE, "webcam", detail + " - slow; close other apps using it")
            else:
                report(OK, "webcam", detail)
        source.release()
    except Exception as exc:
        report(NOTE, "webcam", "could not open (%s)" % str(exc)[:40])


def check_config_and_ports() -> None:
    section("Configuration and ports")
    try:
        from edge.config import load_config
        cfg = load_config()
        report(OK, "config file", os.path.basename(str(cfg.get("_config_path", ""))))
    except Exception as exc:
        report(FAIL, "config file", str(exc)[:60])
        return

    port = int(cfg.server.port)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.6)
    in_use = probe.connect_ex((str(cfg.server.host), port)) == 0
    probe.close()
    if in_use:
        report(NOTE, "server port %d" % port,
               "already in use - a server may be running, or change server.port")
    else:
        report(OK, "server port %d" % port, "free")

    data_dir = os.path.dirname(str(cfg.server.database))
    try:
        os.makedirs(data_dir, exist_ok=True)
        probe_file = os.path.join(data_dir, ".write_test")
        with open(probe_file, "w") as handle:
            handle.write("ok")
        os.remove(probe_file)
        report(OK, "data directory", "writable")
    except Exception as exc:
        report(FAIL, "data directory", "not writable (%s)" % str(exc)[:40])


def check_engine() -> None:
    section("Detection engine")
    try:
        import numpy as np

        from edge.config import load_config
        from edge.fatigue import (STATE_CRITICAL, CalibrationResult,
                                  FatigueEngine)
        from edge.sensors import SensorSample
        from edge.vision import FaceMetrics

        cfg = load_config()
        engine = FatigueEngine(cfg)
        engine.apply_calibration(CalibrationResult(
            ear_baseline=0.30, ear_threshold=0.216, mar_threshold=0.60,
            hr_baseline=72.0, ok=True))

        # Two seconds closed in every five: an unambiguously impaired operator.
        state = None
        for i in range(int(120 * 15)):
            t = i / 15.0
            closed = (t % 5.0) < 2.0
            state = engine.update(
                FaceMetrics(face_found=True, ear=0.08 if closed else 0.30,
                            mar=0.14, pitch=0.0, roll=0.0, quality=1.0,
                            backend="check"),
                SensorSample(heart_rate=72.0, hrv_rmssd=40.0, valid=True,
                             source="check"),
                now=1000.0 + t)

        if state.state == STATE_CRITICAL:
            report(OK, "end-to-end scoring",
                   "drowsy operator -> Critical (score %.0f)" % state.score)
        else:
            report(FAIL, "end-to-end scoring",
                   "expected Critical, got %s (%.0f)" % (state.state, state.score))
    except Exception as exc:
        report(FAIL, "engine self-test", str(exc)[:60])


def main() -> int:
    print("=" * 70)
    print("  Crane Operator Fatigue Monitor - environment check")
    print("=" * 70)

    check_python()
    check_packages()
    check_backend()
    check_camera()
    check_config_and_ports()
    check_engine()

    print("\n" + "=" * 70)
    if _results["fail"]:
        print("  %d blocking problem(s). Fix these, then run this again."
              % _results["fail"])
        print("  Most are solved by:  pip install -r requirements.txt")
        print("=" * 70)
        return 1

    if _results["note"]:
        print("  Ready. %d component(s) running on a documented fallback -"
              % _results["note"])
        print("  the demo works, and the dashboard labels what is simulated.")
    else:
        print("  Ready. Everything is available.")
    print("\n  Start the demo with:   python run_demo.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
