"""
Crane cabin edge node - the main real-time loop.

This is the program that runs on the Raspberry Pi / Jetson bolted inside the
cabin. It follows the numbered algorithm in Section 8 of the abstract:

    1-2   initialise devices, capture the operator baseline
    3-9   per frame: measure EAR, MAR, head tilt and heart rate
    10-11 compute and classify the composite fatigue score
    12-13 drive the local annunciators, escalate critical states upstream
    14    log everything to the control-room database
    15    age counters out of the evaluation window
    16    print a shift summary on exit

Run it with:

    python -m edge.node                       # auto-detect everything
    python -m edge.node --backend synthetic   # no camera needed
    python -m edge.node --headless            # no GUI window (on a Pi)
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque

import cv2
import numpy as np

from .alerts import create_alert_unit
from .config import load_config
from .fatigue import (STATE_CRITICAL, STATE_NO_OPERATOR, Calibrator, FatigueEngine,
                      FatigueState)
from .hud import TraceBuffer, draw_face_overlay, render_dashboard
from .sensors import SensorSample, SimulatedSensorHub, create_sensor_hub
from .telemetry import TelemetryClient
from .vision import VideoSource, create_backend


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


class EdgeNode:
    """Owns every device and runs the detection loop."""

    def __init__(self, cfg, show_window: bool = True):
        self.cfg = cfg
        self.show_window = show_window
        self.running = False

        log("=" * 68)
        log("Crane Operator Fatigue Monitor - edge node %s" % cfg.system.device_id)
        log("=" * 68)

        # --- step 1: initialise devices ----------------------------------
        self.backend = create_backend(cfg.camera.get("backend", "auto"),
                                      cfg.sensors.simulation.get("scenario",
                                                                 "progressive_fatigue"))
        log("vision backend: %s" % self.backend.name)

        self.video = VideoSource(
            cfg.camera.source, int(cfg.camera.width), int(cfg.camera.height),
            bool(cfg.camera.flip_horizontal), bool(cfg.camera.allow_synthetic_fallback))
        log("video source: %s" % self.video.description)

        # A real camera with a synthetic landmark backend would be nonsense,
        # and a synthetic canvas cannot feed a real detector - reconcile them.
        if self.video.is_synthetic and self.backend.name != "synthetic":
            log("no camera present; switching vision backend to synthetic")
            self.backend = create_backend("synthetic",
                                          cfg.sensors.simulation.get("scenario"))

        self.sensors = create_sensor_hub(cfg, logger=log)
        self.sensors.start()

        self.alerts = create_alert_unit(cfg, logger=log)
        self.engine = FatigueEngine(cfg)
        self.calibrator = Calibrator(cfg)

        self.telemetry = TelemetryClient(cfg, logger=log)
        self.telemetry.start()
        log("telemetry: %s -> %s" % (cfg.telemetry.transport, cfg.telemetry.server_url))

        # --- runtime state ------------------------------------------------
        self.traces = {name: TraceBuffer(220)
                       for name in ("ear", "mar", "score", "hr")}
        self.recent_events: deque = deque(maxlen=12)
        self.frame_times: deque = deque(maxlen=30)
        self.last_publish = 0.0
        self.shift_start = time.time()
        self.shift_states: list = []
        self.shift_events: list = []
        self.link_suspended = False        # toggled by [h] to demo buffering
        self.crane_halted = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, duration: float = 0.0) -> None:
        # Opening a camera and probing the sensor buses can take many seconds.
        # Any scripted source must start its shift now, not back when it was
        # constructed, or calibration records an already-tired operator as the
        # normal baseline.
        for device in (self.backend, self.sensors):
            if hasattr(device, "reset_clock"):
                device.reset_clock()

        self.running = True
        started = time.time()
        log("running - press [q] in the window (or Ctrl+C) to stop")

        while self.running:
            loop_start = time.time()

            ok, frame = self.video.read()
            if not ok:
                log("video source ended")
                break

            # --- step 3: capture frame and sensors ------------------------
            face = self.backend.process(frame)
            sample = self.sensors.read()

            # --- step 2: calibration on first run -------------------------
            if self.calibrator.in_progress:
                result = self.calibrator.feed(face, sample)
                if result is not None:
                    self.engine.apply_calibration(result)
                    log("calibration complete: %s" % result.note)
                    if result.ear_baseline:
                        log("  baseline EAR %.3f -> closed-eye threshold %.3f"
                            % (result.ear_baseline, result.ear_threshold))
                    if result.hr_baseline:
                        log("  baseline heart rate %.0f bpm" % result.hr_baseline)
                else:
                    self._render(frame, face, FatigueState(), calibrating=True)
                    if self._handle_keys() is False:
                        break
                    self._pace(loop_start)
                    continue

            # --- steps 4-13: evaluate ------------------------------------
            state = self.engine.update(face, sample)

            # Keep the simulated vitals coherent with what the camera sees.
            if isinstance(self.sensors, SimulatedSensorHub):
                self.sensors.couple_to_observed_state(state.score / 100.0)

            # --- steps 12-13: local annunciation -------------------------
            self.alerts.set_state(state.state, state.score)

            for event in state.events:
                line = "%s  %s" % (time.strftime("%H:%M:%S"), event.message)
                self.recent_events.append((line, event.severity))
                self.shift_events.append(event)
                log("EVENT %-22s %s" % (event.kind, event.message))

            # --- step 14: log upstream -----------------------------------
            self._publish(state)
            self.shift_states.append((state.timestamp, state.score, state.state))

            self._apply_commands()

            for name, value in (("ear", state.ear), ("mar", state.mar),
                                ("score", state.score), ("hr", state.heart_rate)):
                self.traces[name].add(value)

            self._render(frame, face, state)
            if self._handle_keys() is False:
                break

            if duration and (time.time() - started) >= duration:
                log("requested duration reached")
                break

            self._pace(loop_start)

        self.shutdown()

    def _pace(self, loop_start: float) -> None:
        """Hold the loop to the configured frame rate."""
        target = 1.0 / max(1.0, float(self.cfg.system.target_fps))
        elapsed = time.time() - loop_start
        self.frame_times.append(elapsed)
        if elapsed < target:
            time.sleep(target - elapsed)

    @property
    def fps(self) -> float:
        if not self.frame_times:
            return 0.0
        mean = float(np.mean(self.frame_times))
        target = 1.0 / max(1.0, float(self.cfg.system.target_fps))
        return 1.0 / max(mean, target)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _publish(self, state) -> None:
        """
        Routine telemetry is rate-limited; anything interesting goes at once.

        Sending every frame would swamp a GSM link, but delaying a critical
        state change by a second is exactly the wrong economy - so state
        changes and events bypass the rate limiter (algorithm step 13).
        """
        if self.link_suspended:
            return

        now = time.time()
        urgent = bool(state.state_changed or state.events)
        due = (now - self.last_publish) >= float(self.cfg.telemetry.publish_interval_s)
        if not (urgent or due):
            return
        self.last_publish = now

        record = state.to_dict()
        record.update({
            "device_id": self.cfg.system.device_id,
            "crane_id": self.cfg.system.crane_id,
            "site": self.cfg.system.site,
            "operator_id": self.cfg.operator.operator_id,
            "operator_name": self.cfg.operator.name,
            "urgent": urgent,
        })
        self.telemetry.publish(record)

    def _apply_commands(self) -> None:
        """
        Act on supervisor commands - the control-room half of the feedback loop.

        `halt_crane` is deliberately advisory here: it raises the in-cabin alarm
        and flags the state, but this project does not wire an interlock into
        the crane drive. Interlocking a lifting appliance is a functional-safety
        change that needs a rated safety PLC and a site sign-off, not a
        Python process.
        """
        for command in self.telemetry.poll_commands():
            action = str(command.get("action", "")).lower()
            log("supervisor command: %s" % action)

            if action == "acknowledge":
                self.recent_events.append(
                    ("%s  supervisor acknowledged the alert" % time.strftime("%H:%M:%S"),
                     "info"))
            elif action == "recalibrate":
                self.calibrator = Calibrator(self.cfg)
                log("recalibration requested by control room")
            elif action == "reset_counters":
                self.engine.reset_window()
            elif action in ("halt_crane", "pause_operation"):
                self.crane_halted = True
                self.alerts.set_state(STATE_CRITICAL, 100.0)
                self.recent_events.append(
                    ("%s  CONTROL ROOM: HALT / OPERATOR ROTATION REQUESTED"
                     % time.strftime("%H:%M:%S"), "critical"))
            elif action == "resume":
                self.crane_halted = False

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _render(self, frame, face, state, calibrating: bool = False) -> None:
        if not self.show_window:
            self._print_status(state, calibrating)
            return

        draw_face_overlay(frame, face)
        stats = self.telemetry.stats()
        link_text = "%s %s" % (stats["transport"],
                               "connected" if stats["connected"] else "OFFLINE")
        if self.link_suspended:
            link_text = "SUSPENDED (demo)"
        if stats["buffered"]:
            link_text += "  [%d buffered]" % stats["buffered"]

        meta = {
            "crane_id": self.cfg.system.crane_id,
            "device_id": self.cfg.system.device_id,
            "operator_id": self.cfg.operator.operator_id,
            "operator_name": self.cfg.operator.name,
            "fps": self.fps,
            "low_threshold": float(self.cfg.scoring.thresholds.low),
            "high_threshold": float(self.cfg.scoring.thresholds.high),
            "saturation": dict(self.cfg.scoring.saturation),
            "traces": {k: v.array() for k, v in self.traces.items()},
            "recent_events": list(self.recent_events),
            "link_text": link_text,
            "calibrating": calibrating,
            "calibration_progress": self.calibrator.progress(),
            "hr_baseline": self.engine.calibration.hr_baseline,
        }
        canvas = render_dashboard(frame, face, state, meta)
        if self.crane_halted:
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1),
                          (72, 72, 240), 4)
        cv2.imshow("Crane Operator Fatigue Monitor", canvas)

    _last_status = 0.0

    def _print_status(self, state, calibrating: bool) -> None:
        """Single-line status for headless operation."""
        now = time.time()
        if now - self._last_status < 1.0:
            return
        self._last_status = now
        if calibrating:
            sys.stdout.write("\rcalibrating %3.0f%%" % (self.calibrator.progress() * 100))
        else:
            sys.stdout.write(
                "\r%-11s score %5.1f | EAR %s | PERCLOS %3.0f%% | HR %s | %s   "
                % (state.state, state.score,
                   "%.3f" % state.ear if state.ear is not None else " -- ",
                   state.perclos * 100,
                   "%3.0f" % state.heart_rate if state.heart_rate else " --",
                   self.alerts.status_line()))
        sys.stdout.flush()

    def _handle_keys(self):
        if not self.show_window:
            return None
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("c"):
            self.calibrator = Calibrator(self.cfg)
            log("recalibrating")
        elif key == ord("r"):
            self.engine.reset_window()
            log("evaluation-window counters reset")
        elif key == ord("h"):
            self.link_suspended = not self.link_suspended
            log("telemetry link %s (demo of store-and-forward)"
                % ("suspended" if self.link_suspended else "restored"))
        return None

    # ------------------------------------------------------------------
    # Shutdown - algorithm step 16
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self.running = False
        print()
        log("shutting down")
        self.alerts.close()
        self.telemetry.stop()
        self.sensors.stop()
        self.video.release()
        self.backend.close()
        if self.show_window:
            cv2.destroyAllWindows()
        self._print_shift_summary()

    def _print_shift_summary(self) -> None:
        """Step 16: shift summary - fatigue trend, alert count, peak times."""
        if not self.shift_states:
            log("no data captured")
            return

        scores = [s[1] for s in self.shift_states]
        duration = time.time() - self.shift_start
        states = [s[2] for s in self.shift_states]

        counts = {}
        for event in self.shift_events:
            counts[event.kind] = counts.get(event.kind, 0) + 1

        peak_index = int(np.argmax(scores))
        peak_time = time.strftime("%H:%M:%S",
                                  time.localtime(self.shift_states[peak_index][0]))

        print()
        log("-" * 68)
        log("SHIFT SUMMARY - operator %s (%s)"
            % (self.cfg.operator.operator_id, self.cfg.operator.name))
        log("-" * 68)
        log("  duration           : %.1f minutes" % (duration / 60.0))
        log("  samples evaluated  : %d" % len(scores))
        log("  mean fatigue score : %.1f" % float(np.mean(scores)))
        log("  peak fatigue score : %.1f at %s" % (max(scores), peak_time))
        for name in ("Normal", "Warning", "Critical", "NoOperator"):
            n = states.count(name)
            if n:
                log("  time in %-11s: %5.1f %%" % (name, 100.0 * n / len(states)))
        if counts:
            log("  events:")
            for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                log("    %-22s %d" % (kind, n))
        stats = self.telemetry.stats()
        log("  telemetry          : %d sent, %d buffered on disk"
            % (stats["sent"], stats["buffered"]))
        log("-" * 68)
        log("Full shift report: %s/report" % self.cfg.telemetry.server_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crane operator fatigue detection - cabin edge node")
    parser.add_argument("--config", help="path to system_config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override any config key, e.g. --set scoring.thresholds.high=55")
    parser.add_argument("--backend", choices=["auto", "mediapipe", "haar", "synthetic"],
                        help="landmark backend")
    parser.add_argument("--camera", help="camera index or path to a video file")
    parser.add_argument("--sensors", choices=["auto", "serial", "i2c", "simulated"],
                        help="sensor hub driver")
    parser.add_argument("--scenario",
                        choices=["alert", "progressive_fatigue", "microsleep_event"],
                        help="synthetic operator scenario")
    parser.add_argument("--operator", help="operator id signed in to this cabin")
    parser.add_argument("--server", help="control-room server URL")
    parser.add_argument("--headless", action="store_true",
                        help="no GUI window (for a headless Pi)")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="run standalone, do not contact the server")
    parser.add_argument("--no-calibration", action="store_true",
                        help="skip baseline capture and use fallback thresholds")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop automatically after N seconds")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    if args.backend:
        cfg.set_path("camera.backend", args.backend)
    if args.camera is not None:
        try:
            cfg.set_path("camera.source", int(args.camera))
        except ValueError:
            cfg.set_path("camera.source", args.camera)
    if args.sensors:
        cfg.set_path("sensors.driver", args.sensors)
    if args.scenario:
        cfg.set_path("sensors.simulation.scenario", args.scenario)
    if args.operator:
        cfg.set_path("operator.operator_id", args.operator)
    if args.server:
        cfg.set_path("telemetry.server_url", args.server)
    if args.no_telemetry:
        cfg.set_path("telemetry.transport", "none")

    node = EdgeNode(cfg, show_window=not args.headless)
    if args.no_calibration:
        node.engine.apply_calibration(node.calibrator.force_default())

    def on_signal(_signum, _frame):
        node.running = False

    signal.signal(signal.SIGINT, on_signal)

    try:
        node.run(duration=args.duration)
    except KeyboardInterrupt:
        node.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
