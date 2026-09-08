"""
Vision front-end for the fatigue detection node.

Responsibilities (algorithm steps 4-7 of the abstract):
  * grab frames from the cabin camera
  * detect the operator face
  * extract facial landmarks
  * compute Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) and head pose

Three interchangeable landmark backends are provided so the same code runs on
a Jetson / Raspberry Pi with MediaPipe, on a bare OpenCV install, or with no
camera at all:

    MediaPipeBackend   - 478-point FaceMesh, true EAR/MAR + solvePnP head pose
    HaarCascadeBackend - OpenCV-only fallback, geometric openness estimates
    SyntheticBackend   - scripted operator, for demos and automated tests

The backend actually in use is reported in every telemetry packet, so a
fallback measurement is never mistaken for the real thing.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class FaceMetrics:
    """Everything the vision stage extracts from a single frame."""

    face_found: bool = False
    ear: Optional[float] = None          # Eye Aspect Ratio, mean of both eyes
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    mar: Optional[float] = None          # Mouth Aspect Ratio
    pitch: Optional[float] = None        # head nod, degrees (+ve = looking down)
    yaw: Optional[float] = None          # head turn, degrees
    roll: Optional[float] = None         # head lean, degrees
    face_box: Optional[tuple] = None     # (x, y, w, h)
    landmarks: Optional[np.ndarray] = None   # (N, 2) pixel coordinates
    quality: float = 0.0                 # 0..1 confidence in the measurement
    backend: str = "none"
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """
    Soukupova and Cech Eye Aspect Ratio.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Six points ordered [outer, top1, top2, inner, bottom2, bottom1].
    The value is about 0.30 for a wide-open eye and collapses toward 0 when
    the eye closes. It is scale invariant, so it survives the operator moving
    toward or away from the camera.
    """
    p1, p2, p3, p4, p5, p6 = pts
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """
    Mouth Aspect Ratio over the inner lip contour.

        MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (3 * |p1-p5|)

    Points ordered [left corner, top1, top2, top3, right corner,
                    bottom3, bottom2, bottom1].
    A closed mouth sits near 0.05-0.20; a yawn drives it past 0.60.
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = pts
    horizontal = _dist(p1, p5)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p8) + _dist(p3, p7) + _dist(p4, p6)) / (3.0 * horizontal)


# Generic 3D face model (millimetres) used for solvePnP head-pose recovery.
_MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),      # nose tip
    (0.0,  -63.6,  -12.5),      # chin
    (-43.3, 32.7,  -26.0),      # left eye, outer corner
    (43.3,  32.7,  -26.0),      # right eye, outer corner
    (-28.9, -28.9, -24.1),      # left mouth corner
    (28.9,  -28.9, -24.1),      # right mouth corner
], dtype=np.float64)


def solve_head_pose(image_points: np.ndarray, frame_shape) -> tuple:
    """
    Recover (pitch, yaw, roll) in degrees from six 2D landmarks.

    A pinhole camera whose focal length equals the image width is assumed;
    this is the standard approximation used when the cabin camera has not been
    intrinsically calibrated. It is accurate enough to spot a nodding-off
    posture, which is all the fatigue logic needs from vision - the MPU6050
    supplies the precise tilt measurement.
    """
    h, w = frame_shape[:2]
    focal = float(w)
    camera_matrix = np.array([[focal, 0, w / 2.0],
                              [0, focal, h / 2.0],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _tvec = cv2.solvePnP(_MODEL_POINTS_3D,
                                   image_points.astype(np.float64),
                                   camera_matrix, dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-rot_mat[2, 0], sy))
        yaw = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))
        roll = math.degrees(math.atan2(rot_mat[2, 1], rot_mat[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-rot_mat[2, 0], sy))
        yaw = 0.0
        roll = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1]))

    # Fold roll into the +/-90 range a seated human head can actually reach.
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180
    return pitch, yaw, roll


# --------------------------------------------------------------------------
# Backend: MediaPipe FaceMesh (preferred)
# --------------------------------------------------------------------------

# FaceMesh landmark indices, in the point order the EAR/MAR formulas expect.
_MP_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_MP_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_MP_MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
_MP_POSE = [1, 152, 33, 263, 61, 291]


class MediaPipeBackend:
    """
    478-point FaceMesh landmarks - the accurate path.

    Requires `pip install "mediapipe>=0.10,<1.0"`, which ships the classic
    `mediapipe.solutions.face_mesh` API together with its bundled model, so no
    separate model download is needed.
    """

    name = "mediapipe"

    def __init__(self, max_faces: int = 1, min_confidence: float = 0.5):
        import mediapipe as mp          # imported lazily; the fallback still runs
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    @staticmethod
    def available() -> bool:
        try:
            import mediapipe as mp
            return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
        except Exception:
            return False

    def process(self, frame: np.ndarray, now: Optional[float] = None) -> FaceMetrics:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return FaceMetrics(face_found=False, backend=self.name)

        lm = result.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

        ear_r = eye_aspect_ratio(pts[_MP_RIGHT_EYE])
        ear_l = eye_aspect_ratio(pts[_MP_LEFT_EYE])
        mar = mouth_aspect_ratio(pts[_MP_MOUTH])
        pitch, yaw, roll = solve_head_pose(pts[_MP_POSE], frame.shape)

        xs, ys = pts[:, 0], pts[:, 1]
        box = (int(xs.min()), int(ys.min()),
               int(xs.max() - xs.min()), int(ys.max() - ys.min()))

        # Confidence falls off as the head turns away, because a strongly
        # profiled face makes the EAR of the far eye unreliable.
        quality = float(np.clip(1.0 - abs(yaw) / 60.0, 0.15, 1.0))

        return FaceMetrics(
            face_found=True,
            ear=(ear_l + ear_r) / 2.0, ear_left=ear_l, ear_right=ear_r,
            mar=mar, pitch=pitch, yaw=yaw, roll=roll,
            face_box=box, landmarks=pts, quality=quality, backend=self.name,
        )

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Backend: Haar cascades (OpenCV only, no extra dependency)
# --------------------------------------------------------------------------


class HaarCascadeBackend:
    """
    Dependency-free fallback.

    OpenCV ships Haar cascades for faces and eyes but no landmark model, so a
    true 6-point EAR is not available. Instead an *openness ratio* is measured
    directly from the eye region: the iris and lash line are the darkest pixels
    in the patch, and the vertical extent of that dark blob relative to the eye
    width behaves like EAR (about 0.30 open, about 0.10 closed). The same trick
    applied to the mouth region yields a MAR proxy.

    Accuracy is lower than MediaPipe - install mediapipe for production use -
    but the full pipeline, thresholds and alert logic remain exercised.
    """

    name = "haar"

    def __init__(self) -> None:
        base = cv2.data.haarcascades
        self._face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self._eye = cv2.CascadeClassifier(base + "haarcascade_eye_tree_eyeglasses.xml")
        if self._face.empty():
            raise RuntimeError("Could not load the OpenCV frontal-face cascade")

    @staticmethod
    def available() -> bool:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return not cv2.CascadeClassifier(path).empty()

    @staticmethod
    def _dark_blob_ratio(patch: np.ndarray, percentile: float = 22.0) -> float:
        """Vertical extent of the darkest region, divided by the patch width."""
        if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            return 0.0
        patch = cv2.GaussianBlur(patch, (5, 5), 0)
        cutoff = np.percentile(patch, percentile)
        mask = (patch <= cutoff).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return 0.0
        # stats[0] is the background component, so search from index 1.
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        height = float(stats[largest, cv2.CC_STAT_HEIGHT])
        return height / float(patch.shape[1])

    def process(self, frame: np.ndarray, now: Optional[float] = None) -> FaceMetrics:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                            minSize=(90, 90))
        if len(faces) == 0:
            return FaceMetrics(face_found=False, backend=self.name)

        # The largest detection is the operator; anyone further back is a bystander.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # --- eyes: upper band of the face box, split left / right -----------
        eye_band = gray[y + int(0.20 * h): y + int(0.55 * h), x:x + w]
        half = max(1, eye_band.shape[1] // 2)
        ear_r = self._dark_blob_ratio(eye_band[:, :half])
        ear_l = self._dark_blob_ratio(eye_band[:, half:])

        eyes = self._eye.detectMultiScale(eye_band, scaleFactor=1.1, minNeighbors=4,
                                          minSize=(20, 20))
        # The eye cascade only fires on open eyes, so use it to sharpen the estimate.
        bonus = 0.06 if len(eyes) >= 2 else (0.03 if len(eyes) == 1 else -0.03)
        ear_r = max(0.0, ear_r + bonus)
        ear_l = max(0.0, ear_l + bonus)

        # --- mouth: lower third of the face, central 60 % -------------------
        mouth = gray[y + int(0.62 * h): y + h, x + int(0.20 * w): x + int(0.80 * w)]
        mar = self._dark_blob_ratio(mouth, percentile=18.0)

        # --- coarse pose from face-box geometry ------------------------------
        # The box grows taller relative to its width as the head tips forward
        # and the forehead fills the frame; a rough nod indicator only.
        aspect = h / float(w) if w else 1.0
        pitch = float(np.clip((aspect - 1.15) * 90.0, -45.0, 45.0))

        return FaceMetrics(
            face_found=True,
            ear=(ear_l + ear_r) / 2.0, ear_left=ear_l, ear_right=ear_r,
            mar=mar, pitch=pitch, yaw=0.0, roll=0.0,
            face_box=(int(x), int(y), int(w), int(h)), landmarks=None,
            quality=0.55, backend=self.name,
        )

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Backend: scripted synthetic operator (no camera required)
# --------------------------------------------------------------------------


class SyntheticBackend:
    """
    Produces EAR / MAR / pose for a scripted operator who grows progressively
    drowsier, and renders a schematic face so the demo window still shows
    something. Every frame is watermarked SIMULATED so the output is never
    mistaken for a real measurement.
    """

    name = "synthetic"

    def __init__(self, scenario: str = "progressive_fatigue", seed: int = 7):
        self.scenario = scenario
        self._rng = np.random.default_rng(seed)
        self._t0 = time.time()
        self._blink_phase = 0.0
        self._yawn_until = 0.0
        self._microsleep_until = 0.0
        self._next_yawn = 25.0
        self._next_microsleep = 40.0

    @staticmethod
    def available() -> bool:
        return True

    def reset_clock(self, now: Optional[float] = None) -> None:
        """
        Restart the scripted operator's shift.

        Device initialisation (opening a camera, probing sensor buses) can take
        many seconds, and the scripted operator must not have been "working"
        through all of it - otherwise calibration captures an already-fatigued
        baseline. The node calls this once every device is ready.
        """
        self._t0 = time.time() if now is None else now
        self._yawn_until = 0.0
        self._microsleep_until = 0.0
        self._next_yawn = 25.0
        self._next_microsleep = 40.0

    def _fatigue_level(self, elapsed: float) -> float:
        """0 (fresh) .. 1 (exhausted), as a function of elapsed seconds."""
        if self.scenario == "alert":
            return 0.05
        if self.scenario == "microsleep_event":
            return 0.15 if elapsed < 20 else 0.85
        # progressive_fatigue: ramps over about two minutes, then plateaus high
        return float(np.clip(elapsed / 120.0, 0.0, 0.95))

    def process(self, frame: np.ndarray, now: Optional[float] = None) -> FaceMetrics:
        now = time.time() if now is None else now
        elapsed = now - self._t0
        fatigue = self._fatigue_level(elapsed)

        # Blinks get slower and more frequent as fatigue rises. The duty cycle
        # is the fraction of time the eyes are actually shut: a healthy adult
        # blinks ~15 times a minute for ~200 ms, so about 5 % - rising toward
        # 15 % when tired. Overstating it makes an alert operator read as
        # drowsy, because PERCLOS is exactly this quantity.
        blink_rate = 0.25 + 0.9 * fatigue          # blinks per second
        self._blink_phase += blink_rate * (1.0 / 15.0)
        blinking = (self._blink_phase % 1.0) < (0.05 + 0.10 * fatigue)

        # Schedule discrete yawn and microsleep events.
        if elapsed > self._next_yawn:
            self._yawn_until = now + 1.6 + 0.8 * fatigue
            self._next_yawn = elapsed + max(12.0, 45.0 * (1.0 - fatigue) + 8.0)
        if elapsed > self._next_microsleep and fatigue > 0.45:
            self._microsleep_until = now + 0.9 + 2.0 * fatigue
            self._next_microsleep = elapsed + max(10.0, 35.0 * (1.0 - fatigue) + 6.0)

        yawning = now < self._yawn_until
        microsleep = now < self._microsleep_until

        base_ear = 0.315 - 0.05 * fatigue
        if microsleep:
            ear = 0.07 + 0.01 * float(self._rng.normal(0, 1))
        elif blinking:
            ear = 0.10
        else:
            ear = base_ear + 0.012 * float(self._rng.normal(0, 1))
        ear = float(np.clip(ear, 0.03, 0.45))

        if yawning:
            mar = 0.72 + 0.10 * float(self._rng.normal(0, 1))
        else:
            mar = 0.14 + 0.03 * float(self._rng.normal(0, 1))
        mar = float(np.clip(mar, 0.02, 1.1))

        pitch = (18.0 + 14.0 * fatigue) if microsleep else 3.0 * float(self._rng.normal(0, 1))
        roll = 2.0 * float(self._rng.normal(0, 1)) + (9.0 * fatigue if microsleep else 0.0)

        self._render(frame, ear, mar, pitch, fatigue)

        return FaceMetrics(
            face_found=True, ear=ear, ear_left=ear, ear_right=ear, mar=mar,
            pitch=float(pitch), yaw=0.0, roll=float(roll),
            face_box=(frame.shape[1] // 4, frame.shape[0] // 6,
                      frame.shape[1] // 2, int(frame.shape[0] * 0.7)),
            landmarks=None, quality=1.0, backend=self.name,
        )

    def _render(self, frame: np.ndarray, ear: float, mar: float,
                pitch: float, fatigue: float) -> None:
        """Draw a schematic operator so the demo window is not blank."""
        frame[:] = (28, 24, 20)
        h, w = frame.shape[:2]
        cx, cy = w // 2, int(h * 0.46 + pitch * 0.9)

        cv2.ellipse(frame, (cx, cy), (int(w * 0.17), int(h * 0.26)), 0, 0, 360,
                    (150, 172, 196), -1)
        for sign in (-1, 1):
            ex = cx + sign * int(w * 0.075)
            ey = cy - int(h * 0.06)
            eh = max(1, int(ear * h * 0.28))
            cv2.ellipse(frame, (ex, ey), (int(w * 0.045), eh), 0, 0, 360,
                        (35, 32, 30), -1)
        mh = max(2, int(mar * h * 0.20))
        cv2.ellipse(frame, (cx, cy + int(h * 0.13)), (int(w * 0.055), mh), 0, 0, 360,
                    (46, 40, 60), -1)

        # Watermark sits mid-frame, clear of the HUD's own status strips at
        # the top and bottom of the camera view.
        cv2.putText(frame, "SIMULATED INPUT - no camera",
                    (12, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (90, 160, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "synthetic fatigue driver: %5.2f" % fatigue,
                    (12, int(h * 0.96)), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (120, 120, 120), 1, cv2.LINE_AA)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Video source
# --------------------------------------------------------------------------


class VideoSource:
    """Camera, video file, or a blank canvas when neither is available."""

    def __init__(self, source, width: int = 640, height: int = 480,
                 flip: bool = True, allow_synthetic: bool = True):
        self.width, self.height, self.flip = width, height, flip
        self.is_synthetic = False
        self.description = ""
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_file = isinstance(source, str)

        # Windows defaults to the Media Foundation backend, which can take well
        # over a minute to enumerate and open a webcam. DirectShow opens the
        # same camera in under a second, so try it first and keep the default
        # as the fallback. A 100-second startup is not a viable demo.
        apis = [None]
        if not self._is_file and sys.platform.startswith("win"):
            apis = [cv2.CAP_DSHOW, None]

        for api in apis:
            try:
                cap = (cv2.VideoCapture(source) if api is None
                       else cv2.VideoCapture(source, api))
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    ok, _ = cap.read()
                    if ok:
                        self._cap = cap
                        self.description = ("video file %s" % source if self._is_file
                                            else "camera index %s" % source)
                        break
                cap.release()
            except Exception:
                self._cap = None

        if self._cap is None:
            if not allow_synthetic:
                raise RuntimeError("Could not open video source %r" % (source,))
            self.is_synthetic = True
            self.description = "synthetic canvas (no camera detected)"

    def read(self):
        if self._cap is None:
            return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ok, frame = self._cap.read()
        if not ok:
            if self._is_file:            # loop video files for continuous demos
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if self.flip and not self._is_file:
            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def create_backend(prefer: str = "auto", scenario: str = "progressive_fatigue"):
    """
    Pick a landmark backend.

    prefer: auto | mediapipe | haar | synthetic
    'auto' resolves to the most accurate backend that is actually installed.
    """
    prefer = (prefer or "auto").lower()

    if prefer == "synthetic":
        return SyntheticBackend(scenario)
    if prefer == "haar":
        return HaarCascadeBackend()
    if prefer == "mediapipe":
        if not MediaPipeBackend.available():
            raise RuntimeError('mediapipe is not installed '
                               '(pip install "mediapipe>=0.10,<1.0")')
        return MediaPipeBackend()

    if MediaPipeBackend.available():
        try:
            return MediaPipeBackend()
        except Exception:
            pass
    try:
        return HaarCascadeBackend()
    except Exception:
        return SyntheticBackend(scenario)
