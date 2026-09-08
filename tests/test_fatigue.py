"""
Tests for the fatigue scoring engine.

Every test drives the engine on a virtual clock, so a 90-second scenario runs
in milliseconds and the results are exactly reproducible - important, because
these assertions are the evidence that the detection logic behaves the way the
algorithm in the abstract says it does.

    python -m unittest discover tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.config import load_config
from edge.fatigue import (STATE_CRITICAL, STATE_NORMAL, STATE_NO_OPERATOR,
                          STATE_WARNING, Calibrator, CalibrationResult,
                          FatigueEngine, GatedDetector, PerclosTracker,
                          SlidingCounter)
from edge.sensors import SensorSample
from edge.vision import FaceMetrics

OPEN_EAR = 0.30
CLOSED_EAR = 0.08
QUIET_MAR = 0.14
YAWN_MAR = 0.75


def face(ear=OPEN_EAR, mar=QUIET_MAR, found=True, pitch=0.0):
    return FaceMetrics(face_found=found, ear=ear, mar=mar, pitch=pitch,
                       roll=0.0, quality=1.0, backend="test")


def sensors(hr=72.0, tilt=0.0, valid=True, rmssd=40.0):
    return SensorSample(heart_rate=hr, hrv_rmssd=rmssd, tilt_pitch=tilt,
                        tilt_roll=0.0, valid=valid, source="test")


def calibrated_engine(cfg=None, hr_baseline=72.0):
    cfg = cfg or load_config()
    engine = FatigueEngine(cfg)
    engine.apply_calibration(CalibrationResult(
        ear_baseline=OPEN_EAR,
        ear_threshold=OPEN_EAR * float(cfg.calibration.ear_threshold_ratio),
        mar_threshold=float(cfg.calibration.fallback_mar_threshold),
        hr_baseline=hr_baseline, ok=True, note="test"))
    return engine


def run(engine, seconds, face_fn, sensor_fn, start=1000.0, dt=1 / 15.0):
    """Drive the engine for `seconds` of virtual time; return the last state."""
    state = None
    steps = int(seconds / dt)
    for i in range(steps):
        now = start + i * dt
        state = engine.update(face_fn(i * dt), sensor_fn(i * dt), now=now)
    return state


# --------------------------------------------------------------------------


class TestGatedDetector(unittest.TestCase):
    """A threshold crossing must be sustained before it counts."""

    def test_does_not_fire_before_min_duration(self):
        det = GatedDetector(min_duration=1.0, refractory=2.0)
        self.assertFalse(det.update(True, now=0.0))
        self.assertFalse(det.update(True, now=0.5))
        self.assertFalse(det.update(True, now=0.9))

    def test_fires_once_per_excursion(self):
        det = GatedDetector(min_duration=1.0, refractory=2.0)
        det.update(True, now=0.0)
        self.assertTrue(det.update(True, now=1.1))
        # Still held, but a discrete act must not be counted again.
        self.assertFalse(det.update(True, now=1.5))
        self.assertFalse(det.update(True, now=8.0))

    def test_repeat_mode_refires_while_condition_holds(self):
        det = GatedDetector(min_duration=0.4, refractory=0.4, repeat=True)
        det.update(True, now=0.0)
        fires = sum(1 for i in range(1, 30)
                    if det.update(True, now=i * 0.1))
        # 3 s of continuous closure at one count per 0.4 s.
        self.assertGreaterEqual(fires, 6)

    def test_new_excursion_flag_distinguishes_first_fire(self):
        det = GatedDetector(min_duration=0.4, refractory=0.4, repeat=True)
        det.update(True, now=0.0)
        self.assertTrue(det.update(True, now=0.5))
        self.assertTrue(det.is_new_excursion)
        self.assertTrue(det.update(True, now=1.0))
        self.assertFalse(det.is_new_excursion)

    def test_refractory_blocks_immediate_retrigger(self):
        det = GatedDetector(min_duration=0.5, refractory=5.0)
        det.update(True, now=0.0)
        self.assertTrue(det.update(True, now=0.6))
        det.update(False, now=0.7)             # excursion ends
        det.update(True, now=1.0)              # new excursion starts
        self.assertFalse(det.update(True, now=1.7))   # inside refractory
        det.update(False, now=6.0)
        det.update(True, now=6.1)
        self.assertTrue(det.update(True, now=6.8))    # refractory expired


class TestSlidingCounter(unittest.TestCase):

    def test_events_age_out_of_the_window(self):
        counter = SlidingCounter(window=60.0)
        for t in (0.0, 10.0, 20.0):
            counter.add(t)
        self.assertEqual(counter.count(now=30.0), 3)
        self.assertEqual(counter.count(now=75.0), 1)   # only t=20 survives
        self.assertEqual(counter.count(now=200.0), 0)


class TestPerclos(unittest.TestCase):

    def test_measures_closed_time_fraction(self):
        tracker = PerclosTracker(window=60.0)
        for i in range(100):
            tracker.add(closed=(i < 20), dt=0.1, now=i * 0.1)
        self.assertAlmostEqual(tracker.value(), 0.20, places=2)

    def test_coverage_reports_how_full_the_window_is(self):
        tracker = PerclosTracker(window=60.0)
        for i in range(50):
            tracker.add(closed=False, dt=0.1, now=i * 0.1)
        self.assertAlmostEqual(tracker.coverage(), 5.0 / 60.0, places=3)


class TestCalibration(unittest.TestCase):

    def test_threshold_is_derived_from_the_operators_own_baseline(self):
        cfg = load_config()
        cal = Calibrator(cfg)
        for i in range(200):
            cal.feed(face(ear=0.40), sensors(), now=1000.0 + i * 0.05)
        result = cal.feed(face(ear=0.40), sensors(), now=1000.0 + 999)
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.ear_baseline, 0.40, places=2)
        # A wide-eyed operator gets a proportionally higher closed threshold
        # than the textbook 0.21 constant would give them.
        self.assertAlmostEqual(result.ear_threshold,
                               0.40 * float(cfg.calibration.ear_threshold_ratio),
                               places=3)
        self.assertGreater(result.ear_threshold, 0.25)

    def feed_for(self, cal, cfg, ear_fn, frames=200):
        """Run a full calibration window with a scripted EAR sequence."""
        for i in range(frames):
            cal.feed(face(ear=ear_fn(i)), sensors(), now=1000.0 + i * 0.05)
        return cal.feed(face(ear=ear_fn(frames)), sensors(),
                        now=1000.0 + float(cfg.calibration.duration_s) + 1)

    def test_tolerates_ordinary_blinking_during_calibration(self):
        """Roughly 8 % of calibration frames are lost to blinks. That is fine."""
        cfg = load_config()
        result = self.feed_for(Calibrator(cfg), cfg,
                               lambda i: CLOSED_EAR if i % 12 == 0 else 0.32)
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.ear_baseline, 0.32, places=2)

    def test_rejects_an_implausible_baseline(self):
        """
        A camera looking at a half-closed eye must not become "normal".

        If this baseline were accepted, the closure threshold would be set to
        0.08 x 0.72 = 0.058 and no microsleep could ever trip it - the system
        would run all shift reporting Normal while detecting nothing.
        """
        cfg = load_config()
        result = self.feed_for(Calibrator(cfg), cfg, lambda i: 0.08)
        self.assertFalse(result.ok)
        self.assertIsNone(result.ear_baseline)
        self.assertEqual(result.ear_threshold,
                         float(cfg.calibration.fallback_ear_threshold))
        self.assertIn("plausible", result.note)

    def test_rejects_a_baseline_from_an_already_drowsy_operator(self):
        """An operator who dozes through calibration is not a valid reference."""
        cfg = load_config()
        result = self.feed_for(Calibrator(cfg), cfg,
                               lambda i: CLOSED_EAR if (i % 10) < 5 else 0.30)
        self.assertFalse(result.ok)
        self.assertEqual(result.ear_threshold,
                         float(cfg.calibration.fallback_ear_threshold))
        self.assertIn("closed", result.note)

    def test_falls_back_when_the_face_is_never_seen(self):
        cfg = load_config()
        cal = Calibrator(cfg)
        # The first call only starts the calibration clock; the window must
        # actually elapse before a result exists.
        self.assertIsNone(cal.feed(face(found=False), sensors(), now=1000.0))
        result = cal.feed(face(found=False), sensors(),
                          now=1000.0 + float(cfg.calibration.duration_s) + 1)
        self.assertFalse(result.ok)
        self.assertEqual(result.ear_threshold,
                         float(cfg.calibration.fallback_ear_threshold))


class TestFatigueEngine(unittest.TestCase):

    def test_alert_operator_stays_normal(self):
        engine = calibrated_engine()
        state = run(engine, 90, lambda t: face(), lambda t: sensors())
        self.assertEqual(state.state, STATE_NORMAL)
        self.assertLess(state.score, 10.0)
        self.assertEqual(state.counters["drowsiness"], 0)

    def test_normal_blinking_does_not_raise_an_alert(self):
        """A 250 ms blink is shorter than the 400 ms microsleep gate."""
        engine = calibrated_engine()

        def blinking(t):
            # One blink every 4 s, lasting 250 ms.
            return face(ear=CLOSED_EAR if (t % 4.0) < 0.25 else OPEN_EAR)

        state = run(engine, 120, blinking, lambda t: sensors())
        self.assertEqual(state.counters["drowsiness"], 0)
        self.assertEqual(state.state, STATE_NORMAL)

    def test_sustained_eye_closure_escalates_to_critical(self):
        engine = calibrated_engine()

        def drowsy(t):
            # Two seconds closed in every five - a severely impaired operator.
            return face(ear=CLOSED_EAR if (t % 5.0) < 2.0 else OPEN_EAR)

        state = run(engine, 120, drowsy, lambda t: sensors())
        self.assertEqual(state.state, STATE_CRITICAL)
        self.assertGreater(state.score, 60.0)
        self.assertGreater(state.perclos, 0.30)

    def test_yawning_is_counted_once_per_yawn(self):
        engine = calibrated_engine()

        def yawner(t):
            # One 2-second yawn every 20 s -> 3 yawns in 60 s.
            return face(mar=YAWN_MAR if (t % 20.0) < 2.0 else QUIET_MAR)

        state = run(engine, 62, yawner, lambda t: sensors())
        self.assertEqual(state.counters["yawn"], 3)

    def test_head_tilt_from_imu_counts_as_posture(self):
        engine = calibrated_engine()

        def nodding(t):
            return sensors(tilt=35.0 if (t % 15.0) < 2.0 else 2.0)

        state = run(engine, 62, lambda t: face(), nodding)
        self.assertGreaterEqual(state.counters["posture"], 3)

    def test_missing_face_raises_operator_not_visible(self):
        engine = calibrated_engine()
        state = run(engine, 20, lambda t: face(found=False), lambda t: sensors())
        self.assertEqual(state.state, STATE_NO_OPERATOR)
        self.assertIn("operator not visible", " ".join(state.reasons))

    def test_operator_returning_clears_the_alert(self):
        engine = calibrated_engine()
        run(engine, 20, lambda t: face(found=False), lambda t: sensors())
        state = run(engine, 30, lambda t: face(), lambda t: sensors(),
                    start=1000.0 + 20)
        self.assertNotEqual(state.state, STATE_NO_OPERATOR)

    def test_losing_the_heart_rate_sensor_does_not_deflate_the_score(self):
        """
        A missing signal must be dropped from the weighted mean, not scored 0.

        Otherwise unplugging the pulse sensor would make a drowsy operator look
        15 % less fatigued - a failure mode that hides risk.
        """
        def drowsy(t):
            return face(ear=CLOSED_EAR if (t % 5.0) < 2.0 else OPEN_EAR)

        with_hr = run(calibrated_engine(), 90, drowsy, lambda t: sensors())
        without_hr = run(calibrated_engine(), 90, drowsy,
                         lambda t: sensors(valid=False))

        self.assertEqual(without_hr.state, STATE_CRITICAL)
        self.assertGreaterEqual(without_hr.score, with_hr.score - 1.0)

    def test_hysteresis_prevents_state_chatter(self):
        """A score parked exactly on a threshold must not oscillate."""
        cfg = load_config()
        engine = calibrated_engine(cfg)
        low = float(cfg.scoring.thresholds.low)

        # Drive to just above the warning threshold, then hover.
        def borderline(t):
            return face(ear=CLOSED_EAR if (t % 12.0) < 0.6 else OPEN_EAR)

        changes = 0
        state = None
        for i in range(int(180 / (1 / 15.0))):
            now = 1000.0 + i * (1 / 15.0)
            state = engine.update(borderline(i * (1 / 15.0)), sensors(), now=now)
            if state.state_changed:
                changes += 1
        self.assertIsNotNone(state)
        # Without hysteresis and dwell this flips many times a minute.
        self.assertLessEqual(changes, 4)
        self.assertGreaterEqual(low, 0)

    def test_score_is_bounded(self):
        engine = calibrated_engine()

        def awful(t):
            return face(ear=CLOSED_EAR, mar=YAWN_MAR, pitch=60.0)

        state = run(engine, 180, awful, lambda t: sensors(hr=40.0, tilt=50.0))
        self.assertGreaterEqual(state.score, 0.0)
        self.assertLessEqual(state.score, 100.0)
        for value in state.sub_scores.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_perclos_is_ignored_until_the_window_has_enough_data(self):
        """
        Guards the false alarm that used to fire the moment calibration ended.

        With an almost empty window a single blink reads as a huge PERCLOS, so
        the ratio is not consulted until the window is half full.
        """
        engine = calibrated_engine()
        state = None
        for i in range(30):                   # 2 seconds of data only
            now = 1000.0 + i * (1 / 15.0)
            closed = i < 6                    # a 400 ms blink at the start
            state = engine.update(face(ear=CLOSED_EAR if closed else OPEN_EAR),
                                  sensors(), now=now)
        self.assertGreater(state.perclos, 0.15)      # ratio is high...
        self.assertEqual(state.state, STATE_NORMAL)  # ...but correctly ignored

    def test_reset_window_clears_counters(self):
        engine = calibrated_engine()

        def drowsy(t):
            return face(ear=CLOSED_EAR if (t % 5.0) < 2.0 else OPEN_EAR)

        run(engine, 90, drowsy, lambda t: sensors())
        engine.reset_window()
        state = engine.update(face(), sensors(), now=1200.0)
        self.assertEqual(state.counters["drowsiness"], 0)
        self.assertEqual(state.perclos, 0.0)

    def test_state_change_events_are_emitted(self):
        engine = calibrated_engine()

        def drowsy(t):
            return face(ear=CLOSED_EAR if (t % 5.0) < 2.0 else OPEN_EAR)

        kinds = []
        for i in range(int(120 / (1 / 15.0))):
            now = 1000.0 + i * (1 / 15.0)
            state = engine.update(drowsy(i * (1 / 15.0)), sensors(), now=now)
            kinds.extend(e.kind for e in state.events)

        self.assertIn("state_change", kinds)
        self.assertIn("microsleep", kinds)

    def test_one_long_microsleep_logs_one_event_but_counts_many(self):
        engine = calibrated_engine()
        events = 0
        for i in range(int(4.0 / (1 / 15.0))):        # 4 s of closed eyes
            now = 1000.0 + i * (1 / 15.0)
            state = engine.update(face(ear=CLOSED_EAR), sensors(), now=now)
            events += sum(1 for e in state.events if e.kind == "microsleep")

        self.assertEqual(events, 1, "the control-room feed must not be flooded")
        self.assertGreater(state.counters["drowsiness"], 1,
                           "a longer closure must still score higher")


if __name__ == "__main__":
    unittest.main(verbosity=2)
