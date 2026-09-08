"""
Tests for the signal-processing primitives: EAR, MAR, head pose, heart-rate
analysis and accelerometer tilt.

These are the formulas quoted in the abstract, so they get checked against
hand-constructed geometry with known answers rather than against a recording.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.sensors import HeartRateAnalyser, tilt_from_accel
from edge.vision import (HaarCascadeBackend, MediaPipeBackend, SyntheticBackend,
                         create_backend, eye_aspect_ratio, mouth_aspect_ratio,
                         solve_head_pose)


class TestEyeAspectRatio(unittest.TestCase):

    @staticmethod
    def eye(width: float, height: float) -> np.ndarray:
        """Six landmarks for an eye of the given width and lid separation."""
        half = height / 2.0
        return np.array([
            [0.0, 0.0],                       # p1 outer corner
            [width * 0.25, -half],            # p2 upper lid
            [width * 0.75, -half],            # p3 upper lid
            [width, 0.0],                     # p4 inner corner
            [width * 0.75, half],             # p5 lower lid
            [width * 0.25, half],             # p6 lower lid
        ])

    def test_open_eye_is_near_the_published_value(self):
        # A typical open eye is about 8 units wide and 2.4 tall -> EAR ~0.30
        self.assertAlmostEqual(eye_aspect_ratio(self.eye(8.0, 2.4)), 0.30, places=2)

    def test_closed_eye_collapses_toward_zero(self):
        self.assertLess(eye_aspect_ratio(self.eye(8.0, 0.3)), 0.05)

    def test_ear_is_scale_invariant(self):
        """The operator leaning toward the camera must not change the EAR."""
        near = eye_aspect_ratio(self.eye(16.0, 4.8))
        far = eye_aspect_ratio(self.eye(4.0, 1.2))
        self.assertAlmostEqual(near, far, places=6)

    def test_degenerate_input_does_not_raise(self):
        self.assertEqual(eye_aspect_ratio(np.zeros((6, 2))), 0.0)


class TestMouthAspectRatio(unittest.TestCase):

    @staticmethod
    def mouth(width: float, opening: float) -> np.ndarray:
        half = opening / 2.0
        return np.array([
            [0.0, 0.0],                        # p1 left corner
            [width * 0.25, -half],             # p2 upper
            [width * 0.50, -half],             # p3 upper
            [width * 0.75, -half],             # p4 upper
            [width, 0.0],                      # p5 right corner
            [width * 0.75, half],              # p6 lower
            [width * 0.50, half],              # p7 lower
            [width * 0.25, half],              # p8 lower
        ])

    def test_closed_mouth_is_low(self):
        self.assertLess(mouth_aspect_ratio(self.mouth(10.0, 1.0)), 0.15)

    def test_yawn_exceeds_the_default_threshold(self):
        # A yawn opens the jaw to roughly two-thirds of the mouth width.
        self.assertGreater(mouth_aspect_ratio(self.mouth(10.0, 6.5)), 0.60)

    def test_degenerate_input_does_not_raise(self):
        self.assertEqual(mouth_aspect_ratio(np.zeros((8, 2))), 0.0)


class TestHeadPose(unittest.TestCase):

    def test_frontal_face_reports_near_zero_pose(self):
        # Landmarks laid out symmetrically about the image centre.
        points = np.array([
            [320.0, 240.0],    # nose tip
            [320.0, 310.0],    # chin
            [270.0, 205.0],    # left eye outer
            [370.0, 205.0],    # right eye outer
            [290.0, 275.0],    # left mouth corner
            [350.0, 275.0],    # right mouth corner
        ])
        pitch, yaw, roll = solve_head_pose(points, (480, 640, 3))
        self.assertLess(abs(yaw), 15.0)
        self.assertLess(abs(roll), 15.0)

    def test_returns_three_finite_angles(self):
        points = np.array([[300.0, 250.0], [305.0, 320.0], [255.0, 215.0],
                           [355.0, 210.0], [275.0, 285.0], [335.0, 283.0]])
        angles = solve_head_pose(points, (480, 640, 3))
        self.assertEqual(len(angles), 3)
        for angle in angles:
            self.assertTrue(math.isfinite(angle))


class TestHeartRateAnalyser(unittest.TestCase):

    def test_computes_rate_from_beat_timestamps(self):
        analyser = HeartRateAnalyser()
        for i in range(20):
            analyser.add_beat(i * 0.8)         # 800 ms RR -> 75 bpm
        self.assertAlmostEqual(analyser.heart_rate(), 75.0, places=1)

    def test_perfectly_regular_beats_have_zero_variability(self):
        analyser = HeartRateAnalyser()
        for i in range(20):
            analyser.add_beat(i * 0.8)
        self.assertAlmostEqual(analyser.rmssd(), 0.0, places=6)

    def test_rmssd_rises_with_beat_to_beat_variation(self):
        analyser = HeartRateAnalyser()
        t = 0.0
        for i in range(20):
            t += 0.8 + (0.05 if i % 2 else -0.05)
            analyser.add_beat(t)
        self.assertGreater(analyser.rmssd(), 50.0)

    def test_motion_artefacts_are_rejected(self):
        """An impossible RR interval must not corrupt the average."""
        analyser = HeartRateAnalyser()
        for i in range(10):
            analyser.add_beat(i * 0.8)         # last real beat at t = 7.2 s
        before = analyser.heart_rate()

        # A double-detection 50 ms after the previous beat implies 1200 bpm.
        analyser.add_beat(7.25)
        self.assertAlmostEqual(analyser.heart_rate(), before, places=6)

        # And a two-minute gap (sensor lifted off the finger) is equally bogus.
        analyser.add_beat(127.25)
        self.assertAlmostEqual(analyser.heart_rate(), before, places=6)

    def test_reports_nothing_until_enough_beats(self):
        analyser = HeartRateAnalyser()
        self.assertIsNone(analyser.heart_rate())
        analyser.add_beat(0.0)
        self.assertIsNone(analyser.heart_rate())


class TestTiltFromAccel(unittest.TestCase):

    def test_upright_head_is_level(self):
        pitch, roll, magnitude = tilt_from_accel(0.0, 0.0, 1.0)
        self.assertAlmostEqual(pitch, 0.0, places=3)
        self.assertAlmostEqual(roll, 0.0, places=3)
        self.assertAlmostEqual(magnitude, 0.0, places=3)

    def test_forward_nod_produces_pitch(self):
        angle = math.radians(30.0)
        pitch, _roll, magnitude = tilt_from_accel(-math.sin(angle), 0.0,
                                                  math.cos(angle))
        self.assertAlmostEqual(pitch, 30.0, places=1)
        self.assertAlmostEqual(magnitude, 30.0, places=1)

    def test_sideways_lean_produces_roll(self):
        angle = math.radians(25.0)
        _pitch, roll, magnitude = tilt_from_accel(0.0, math.sin(angle),
                                                  math.cos(angle))
        self.assertAlmostEqual(roll, 25.0, places=1)
        self.assertAlmostEqual(magnitude, 25.0, places=1)


class TestVisionBackends(unittest.TestCase):

    def test_synthetic_backend_is_always_available(self):
        self.assertTrue(SyntheticBackend.available())

    def test_auto_selection_returns_a_usable_backend(self):
        backend = create_backend("auto")
        self.assertIn(backend.name, ("mediapipe", "haar", "synthetic"))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = backend.process(frame)
        self.assertIn(metrics.backend, ("mediapipe", "haar", "synthetic"))
        backend.close()

    def test_synthetic_operator_gets_drowsier_over_time(self):
        backend = create_backend("synthetic", "progressive_fatigue")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        base = backend._t0

        early = [backend.process(frame, now=base + t / 15.0).ear
                 for t in range(150)]
        late = [backend.process(frame, now=base + 150.0 + t / 15.0).ear
                for t in range(150)]

        # Eyes spend measurably more time closed once fatigue has ramped up.
        self.assertLess(float(np.mean(late)), float(np.mean(early)))

    def test_backends_report_no_face_on_a_blank_frame(self):
        """A blank frame contains no operator, and must be reported as such."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for cls in (MediaPipeBackend, HaarCascadeBackend):
            if not cls.available():
                continue
            backend = cls()
            self.assertFalse(backend.process(frame).face_found,
                             "%s hallucinated a face" % cls.name)
            backend.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
