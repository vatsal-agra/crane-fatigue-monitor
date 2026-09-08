"""
Generate a full simulated shift of history.

The live demo shows the system detecting fatigue in real time, but a reviewer
also wants to see the *reporting* side, and that needs hours of history rather
than the two minutes a demo lasts. This tool synthesises a plausible shift for
several cabins and writes it straight into the control-room database.

Every record it writes is tagged `sensor_source="simulated"`, so simulated
history is distinguishable from real captures in the database and on screen.

    python -m tools.simulate_shift                    # 3 cranes, last 8 hours
    python -m tools.simulate_shift --hours 12 --cranes 4
    python -m tools.simulate_shift --reset            # clear first

The fatigue model applied here is not the detector - it is a *driver* that
produces plausible physiological inputs, which are then scored by the real
FatigueEngine. So the states and alerts in the generated history come from the
same code that runs live, not from a lookup table.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.config import load_config                      # noqa: E402
from edge.fatigue import CalibrationResult, FatigueEngine  # noqa: E402
from edge.sensors import SensorSample                     # noqa: E402
from edge.vision import FaceMetrics                       # noqa: E402
from server.database import Database                      # noqa: E402

CRANES = [
    ("CRANE-01-CABIN", "TOWER-CRANE-01", "Site A - Block 3", "OP-1043", "R. Kumar"),
    ("CRANE-02-CABIN", "TOWER-CRANE-02", "Site A - Block 5", "OP-2210", "S. Fernandes"),
    ("CRANE-03-CABIN", "CRAWLER-CRANE-01", "Site A - Yard", "OP-3378", "A. Bose"),
    ("CRANE-04-CABIN", "TOWER-CRANE-03", "Site B - North", "OP-4102", "M. Iqbal"),
]


def circadian_pressure(hour_of_day: float) -> float:
    """
    Baseline sleepiness by clock hour, 0..1.

    Two-process model, simplified: the well-documented post-lunch dip around
    14:00-15:00 and the deep trough in the small hours. This is what makes the
    generated report show fatigue-prone *hours* rather than uniform noise.
    """
    early = math.exp(-((hour_of_day - 3.5) ** 2) / 6.0)        # 02:00-05:00 trough
    afternoon = 0.65 * math.exp(-((hour_of_day - 14.5) ** 2) / 3.0)
    return min(1.0, early + afternoon)


def operator_profile(index: int, rng: random.Random) -> dict:
    """Give each operator a different susceptibility, so reports differentiate."""
    return {
        "susceptibility": rng.uniform(0.55, 1.35),
        "baseline_hr": rng.uniform(64.0, 82.0),
        "baseline_ear": rng.uniform(0.27, 0.34),
    }


def simulate(db: Database, cfg, hours: float, cranes: int, step_s: float,
             publish_every: int = 4, seed: int = 42) -> int:
    """
    Mirror the real architecture: evaluate on a fine grid, publish on a coarse
    one.

    The engine gates eye closure at 0.4 s and measures PERCLOS as a time
    fraction, so feeding it one sample every two seconds would both miss
    microsleeps and badly distort PERCLOS. So the driver steps at `step_s`
    (0.5 s by default) and only every `publish_every`-th evaluated state is
    written to the database - which is precisely what the cabin unit does when
    it evaluates at 15 fps and publishes at 1 Hz.
    """
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    now = time.time()
    start = now - hours * 3600.0
    total_written = 0

    for index in range(min(cranes, len(CRANES))):
        device_id, crane_id, site, operator_id, operator_name = CRANES[index]
        profile = operator_profile(index, rng)

        engine = FatigueEngine(cfg)
        engine.apply_calibration(CalibrationResult(
            ear_baseline=profile["baseline_ear"],
            ear_threshold=profile["baseline_ear"] * float(cfg.calibration.ear_threshold_ratio),
            mar_threshold=float(cfg.calibration.fallback_mar_threshold),
            hr_baseline=profile["baseline_hr"],
            ok=True, note="simulated baseline"))

        batch: list = []
        sample_index = 0
        t = start
        yawn_until = 0.0
        microsleep_until = 0.0
        away_until = 0.0
        next_break = start + rng.uniform(1800, 4200)

        while t < now:
            elapsed_h = (t - start) / 3600.0
            hour_of_day = time.localtime(t).tm_hour + time.localtime(t).tm_min / 60.0

            # Fatigue pressure: time on task + circadian rhythm + individual factor
            pressure = (0.42 * min(1.0, elapsed_h / max(1e-6, hours))
                        + 0.58 * circadian_pressure(hour_of_day))
            pressure *= profile["susceptibility"]

            # A break resets time-on-task pressure for a while.
            if t > next_break:
                away_until = t + rng.uniform(240, 600)      # operator leaves the seat
                next_break = t + rng.uniform(2700, 5400)
                start += rng.uniform(600, 1500)             # partial recovery
            fatigue = float(np.clip(pressure, 0.0, 1.0))

            if t < away_until:
                face = FaceMetrics(face_found=False, backend="mediapipe")
                sample = SensorSample(heart_rate=None, valid=False, source="simulated")
            else:
                # Rates are expressed per minute and converted to a per-step
                # probability, so changing --step does not change the physiology.
                # A markedly drowsy operator shows several microsleeps a minute.
                microsleep_per_min = 4.2 * (fatigue ** 1.6)
                yawn_per_min = 1.9 * (0.15 + fatigue) ** 1.3
                if (t > microsleep_until
                        and rng.random() < microsleep_per_min * step_s / 60.0):
                    microsleep_until = t + rng.uniform(0.9, 3.4)
                if t > yawn_until and rng.random() < yawn_per_min * step_s / 60.0:
                    yawn_until = t + rng.uniform(1.2, 3.0)

                microsleep = t < microsleep_until
                yawning = t < yawn_until

                # Each generated sample stands for `step_s` of wall time, so a
                # blink must be emitted with the probability that the eyes are
                # shut *at that instant*, not once per blink. Modelling it the
                # other way stretches a 200 ms blink into a 500 ms closure and
                # inflates PERCLOS roughly threefold - which then reads as
                # drowsiness in an operator who is merely blinking normally.
                blinks_per_min = 15.0 + 10.0 * fatigue
                blink_duration = 0.20 + 0.15 * fatigue
                blink = rng.random() < blinks_per_min * blink_duration / 60.0

                base_ear = profile["baseline_ear"] - 0.045 * fatigue
                if microsleep:
                    ear = rng.uniform(0.05, 0.10)
                elif blink:
                    ear = rng.uniform(0.09, 0.14)
                else:
                    ear = base_ear + float(nprng.normal(0, 0.012))

                mar = (rng.uniform(0.62, 0.95) if yawning
                       else 0.14 + float(nprng.normal(0, 0.03)))

                pitch = (rng.uniform(20, 38) if microsleep
                         else float(nprng.normal(0, 3.5)) + 5.0 * fatigue)

                hr = (profile["baseline_hr"] - 15.0 * fatigue
                      + 2.2 * math.sin(2 * math.pi * (t - start) / 4.5)
                      + float(nprng.normal(0, 1.4)))
                rmssd = max(6.0, 46.0 - 30.0 * fatigue + float(nprng.normal(0, 4.0)))

                face = FaceMetrics(face_found=True, ear=float(np.clip(ear, 0.03, 0.45)),
                                   mar=float(np.clip(mar, 0.02, 1.1)),
                                   pitch=pitch, yaw=0.0, roll=float(nprng.normal(0, 3)),
                                   quality=1.0, backend="mediapipe")
                sample = SensorSample(heart_rate=float(np.clip(hr, 42, 130)),
                                      hrv_rmssd=rmssd, tilt_pitch=pitch,
                                      tilt_roll=float(nprng.normal(0, 3)),
                                      valid=True, source="simulated")

            state = engine.update(face, sample, now=t)

            # Always publish a state change or an event, exactly as the edge
            # node does - otherwise decimation would silently drop alerts.
            urgent = bool(state.state_changed or state.events)
            if urgent or sample_index % publish_every == 0:
                record = state.to_dict()
                record.update({
                    "device_id": device_id, "crane_id": crane_id, "site": site,
                    "operator_id": operator_id, "operator_name": operator_name,
                })
                batch.append(record)

            if len(batch) >= 500:
                total_written += db.ingest(batch)
                batch = []
            sample_index += 1
            t += step_s

        if batch:
            total_written += db.ingest(batch)

        print("  %-16s %-18s operator %s" % (device_id, crane_id, operator_id))

    return total_written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate simulated shift history")
    parser.add_argument("--config")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--cranes", type=int, default=3)
    parser.add_argument("--step", type=float, default=0.5,
                        help="seconds between evaluated samples (detection grid)")
    parser.add_argument("--publish-every", type=int, default=4,
                        help="write every Nth evaluated sample to the database")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true",
                        help="clear existing data before generating")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    db = Database(str(cfg.server.database))

    if args.reset:
        db.reset()
        print("cleared existing data")

    print("generating %.0f hours of history for %d cabin units..."
          % (args.hours, args.cranes))
    started = time.time()
    written = simulate(db, cfg, args.hours, args.cranes, args.step,
                       args.publish_every, args.seed)
    print("wrote %d telemetry records in %.1f s" % (written, time.time() - started))

    report = db.shift_report(hours=args.hours)
    print()
    print("  mean fatigue score : %.1f" % report["mean_score"])
    print("  peak fatigue score : %.1f" % report["peak_score"])
    print("  escalated alerts   : %d" % report["total_alerts"])
    print("  state distribution : %s" % report["state_distribution"])
    print()
    print("Open http://%s:%d/report to view it."
          % (cfg.server.host, int(cfg.server.port)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
