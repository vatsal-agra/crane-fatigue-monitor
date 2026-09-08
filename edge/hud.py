"""
Operator-facing heads-up display for the edge node.

This is the window a reviewer actually watches, so it shows the whole chain at
once: the raw video, the landmarks the detector is using, the two aspect ratios
against their live thresholds, the four indicator counters, the composite score
and the resulting alert state.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .fatigue import STATE_CRITICAL, STATE_NO_OPERATOR, STATE_NORMAL, STATE_WARNING

# BGR palette, tuned for a dark control-room look.
BG = (26, 22, 20)
PANEL = (42, 36, 32)
GRID = (62, 55, 50)
TEXT = (226, 226, 226)
MUTED = (150, 146, 142)
GREEN = (110, 210, 130)
YELLOW = (60, 200, 245)
RED = (72, 72, 240)
MAGENTA = (220, 110, 220)
BLUE = (235, 170, 90)

STATE_COLOURS = {
    STATE_NORMAL: GREEN,
    STATE_WARNING: YELLOW,
    STATE_CRITICAL: RED,
    STATE_NO_OPERATOR: MAGENTA,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX

# MediaPipe landmark rings, for drawing the detected eye and mouth contours.
_EYE_RINGS = [
    [33, 160, 158, 133, 153, 144],
    [362, 385, 387, 263, 373, 380],
]
_MOUTH_RING = [78, 81, 13, 311, 308, 402, 14, 178]


class TraceBuffer:
    """Fixed-length history of a scalar signal, for the sparkline plots."""

    def __init__(self, length: int = 220):
        self._values: deque = deque(maxlen=length)

    def add(self, value) -> None:
        self._values.append(float("nan") if value is None else float(value))

    def array(self) -> np.ndarray:
        return np.array(self._values, dtype=float)

    def __len__(self) -> int:
        return len(self._values)


def _text(img, s, org, scale=0.45, colour=TEXT, thickness=1):
    cv2.putText(img, s, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def _panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, 1)
    if title:
        _text(img, title, (x + 10, y + 18), 0.42, MUTED)


def draw_face_overlay(frame, face) -> None:
    """Draw the detection result onto the raw camera frame, in place."""
    if not face.face_found:
        _text(frame, "NO FACE DETECTED", (14, 30), 0.7, MAGENTA, 2)
        return

    if face.face_box:
        x, y, w, h = face.face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 140, 90), 1)

    if face.landmarks is not None and len(face.landmarks) > 400:
        for ring in _EYE_RINGS:
            pts = face.landmarks[ring].astype(np.int32)
            cv2.polylines(frame, [pts], True, GREEN, 1, cv2.LINE_AA)
        pts = face.landmarks[_MOUTH_RING].astype(np.int32)
        cv2.polylines(frame, [pts], True, BLUE, 1, cv2.LINE_AA)


def _draw_trace(img, x, y, w, h, values, threshold, colour, label,
                lo=None, hi=None):
    """Sparkline with a dashed threshold line and the current value."""
    _panel(img, x, y, w, h)
    _text(img, label, (x + 8, y + 15), 0.4, MUTED)

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        _text(img, "waiting for data", (x + 8, y + h // 2), 0.4, MUTED)
        return

    lo = float(np.nanmin(finite)) if lo is None else lo
    hi = float(np.nanmax(finite)) if hi is None else hi
    lo = min(lo, threshold) if threshold is not None else lo
    hi = max(hi, threshold) if threshold is not None else hi
    span = max(1e-6, hi - lo)

    plot_y, plot_h = y + 22, h - 30
    plot_x, plot_w = x + 8, w - 16

    def to_px(value):
        return int(plot_y + plot_h - (value - lo) / span * plot_h)

    if threshold is not None:
        ty = to_px(threshold)
        for dash_x in range(plot_x, plot_x + plot_w, 8):
            cv2.line(img, (dash_x, ty), (min(dash_x + 4, plot_x + plot_w), ty),
                     (90, 90, 130), 1)
        # Only label the threshold where there is room for it: when the line
        # sits up against the panel title or down on the axis, the label would
        # overprint them and both become unreadable.
        if y + 30 < ty < y + h - 12:
            _text(img, "%.2f" % threshold, (plot_x + 3, ty - 4), 0.34, (120, 120, 170))

    step = plot_w / float(max(1, len(values) - 1))
    points = []
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        points.append((int(plot_x + i * step), to_px(value)))
    if len(points) > 1:
        cv2.polylines(img, [np.array(points, np.int32)], False, colour, 1, cv2.LINE_AA)

    current = finite[-1]
    _text(img, "%.3f" % current, (x + w - 58, y + 15), 0.44, colour)


def _draw_gauge(img, x, y, w, h, score, low, high, state):
    """Horizontal fatigue-score gauge with the Normal/Warning/Critical bands."""
    _panel(img, x, y, w, h, "FATIGUE SCORE")
    bar_x, bar_y = x + 12, y + 30
    bar_w, bar_h = w - 24, 20

    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (34, 30, 28), -1)
    # Zone shading
    low_px = int(bar_w * low / 100.0)
    high_px = int(bar_w * high / 100.0)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + low_px, bar_y + bar_h), (40, 60, 44), -1)
    cv2.rectangle(img, (bar_x + low_px, bar_y), (bar_x + high_px, bar_y + bar_h),
                  (40, 62, 74), -1)
    cv2.rectangle(img, (bar_x + high_px, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 34, 62), -1)

    fill = int(bar_w * max(0.0, min(100.0, score)) / 100.0)
    colour = STATE_COLOURS.get(state, GREEN)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), colour, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GRID, 1)

    _text(img, "%.0f" % score, (bar_x + bar_w + -46, bar_y + 15), 0.6, TEXT, 2)
    _text(img, "0", (bar_x, bar_y + bar_h + 14), 0.34, MUTED)
    _text(img, "%d" % low, (bar_x + low_px - 6, bar_y + bar_h + 14), 0.34, MUTED)
    _text(img, "%d" % high, (bar_x + high_px - 6, bar_y + bar_h + 14), 0.34, MUTED)
    _text(img, "100", (bar_x + bar_w - 16, bar_y + bar_h + 14), 0.34, MUTED)


def _draw_counters(img, x, y, w, h, state_obj, saturation):
    _panel(img, x, y, w, h, "INDICATOR COUNTERS  (sliding 60 s window)")
    rows = [("Drowsiness / microsleep", "drowsiness", GREEN),
            ("Yawns", "yawn", BLUE),
            ("Nod-off posture", "posture", YELLOW),
            ("Heart-rate anomaly", "vitals", RED)]
    row_y = y + 34
    for label, key, colour in rows:
        count = state_obj.counters.get(key, 0)
        sub = state_obj.sub_scores.get(key, 0.0)
        cap = int(saturation.get(key, 3))
        _text(img, label, (x + 12, row_y + 4), 0.4, TEXT)
        _text(img, "%d/%d" % (count, cap), (x + 196, row_y + 4), 0.4, MUTED)

        bx, bw = x + 240, w - 258
        cv2.rectangle(img, (bx, row_y - 7), (bx + bw, row_y + 3), (34, 30, 28), -1)
        cv2.rectangle(img, (bx, row_y - 7),
                      (bx + int(bw * min(1.0, sub)), row_y + 3), colour, -1)
        row_y += 24


def render_dashboard(frame, face, state_obj, meta) -> np.ndarray:
    """
    Compose the full HUD.

    frame     : the raw camera frame (overlay already drawn)
    face      : FaceMetrics for this frame
    state_obj : FatigueState from the engine
    meta      : dict of runtime info (fps, link status, operator, traces, ...)
    """
    W, H = 1180, 660
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    # ---- header ---------------------------------------------------------
    cv2.rectangle(canvas, (0, 0), (W, 52), (36, 31, 28), -1)
    _text(canvas, "CRANE OPERATOR FATIGUE MONITOR", (18, 24), 0.62, TEXT, 2)
    _text(canvas, "%s   |   %s   |   operator %s (%s)"
          % (meta.get("crane_id", "-"), meta.get("device_id", "-"),
             meta.get("operator_id", "-"), meta.get("operator_name", "-")),
          (18, 42), 0.4, MUTED)

    state = state_obj.state
    colour = STATE_COLOURS.get(state, GREEN)
    badge_w = 210
    cv2.rectangle(canvas, (W - badge_w - 16, 10), (W - 16, 44), colour, -1)
    label = "NO OPERATOR" if state == STATE_NO_OPERATOR else state.upper()
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.68, 2)
    _text(canvas, label, (W - badge_w - 16 + (badge_w - tw) // 2, 35), 0.68, (20, 18, 16), 2)

    # ---- camera view ----------------------------------------------------
    view_w, view_h = 520, 368
    strip_h = 22
    view = cv2.resize(frame, (view_w, view_h))
    canvas[64 + strip_h:64 + strip_h + view_h, 16:16 + view_w] = view
    cv2.rectangle(canvas, (16, 64), (16 + view_w, 64 + strip_h), PANEL, -1)
    cv2.rectangle(canvas, (16, 64), (16 + view_w, 64 + strip_h + view_h), GRID, 1)
    _text(canvas, "vision: %s   %.1f fps   quality %.2f"
          % (state_obj.vision_backend, meta.get("fps", 0.0), face.quality),
          (24, 79), 0.4, (200, 200, 200))
    view_h += strip_h

    if meta.get("calibrating"):
        pct = meta.get("calibration_progress", 0.0)
        cv2.rectangle(canvas, (16, 64), (16 + view_w, 64 + view_h), (0, 0, 0), -1)
        _text(canvas, "CALIBRATING OPERATOR BASELINE", (56, 200), 0.62, TEXT, 2)
        _text(canvas, "Sit normally and look at the camera", (86, 226), 0.44, MUTED)
        cv2.rectangle(canvas, (56, 250), (56 + 400, 268), (40, 36, 32), -1)
        cv2.rectangle(canvas, (56, 250), (56 + int(400 * pct), 268), BLUE, -1)

    # ---- right column ---------------------------------------------------
    rx, rw = 552, W - 552 - 16
    _draw_gauge(canvas, rx, 64, rw, 78, state_obj.score,
                meta.get("low_threshold", 30), meta.get("high_threshold", 60), state)
    _draw_counters(canvas, rx, 150, rw, 128, state_obj, meta.get("saturation", {}))

    traces = meta.get("traces", {})
    half = (rw - 10) // 2
    _draw_trace(canvas, rx, 286, half, 120, traces.get("ear", np.array([])),
                state_obj.ear_threshold, GREEN, "EYE ASPECT RATIO (EAR)", lo=0.0, hi=0.45)
    _draw_trace(canvas, rx + half + 10, 286, half, 120, traces.get("mar", np.array([])),
                state_obj.mar_threshold, BLUE, "MOUTH ASPECT RATIO (MAR)", lo=0.0, hi=1.0)
    _draw_trace(canvas, rx, 414, half, 120, traces.get("score", np.array([])),
                meta.get("high_threshold", 60), YELLOW, "FATIGUE SCORE", lo=0.0, hi=100.0)
    _draw_trace(canvas, rx + half + 10, 414, half, 120, traces.get("hr", np.array([])),
                meta.get("hr_baseline"), RED, "HEART RATE (bpm)")

    # ---- vitals / status strip -----------------------------------------
    _panel(canvas, 16, 64 + view_h + 10, view_w, 176, "SENSOR HUB  /  LINK")
    sy = 64 + view_h + 50
    hrv = state_obj.hrv_rmssd
    rows = [
        ("Heart rate", "%.0f bpm" % state_obj.heart_rate if state_obj.heart_rate else "--"),
        ("HRV (RMSSD)", "%.0f ms" % hrv if hrv else "--"),
        ("Head tilt", "%.1f deg" % state_obj.tilt if state_obj.tilt is not None else "--"),
        ("PERCLOS", "%.0f %%" % (state_obj.perclos * 100)),
        ("Sensor source", state_obj.sensor_source),
        ("Telemetry", meta.get("link_text", "--")),
    ]
    for label, value in rows:
        _text(canvas, label, (28, sy), 0.4, MUTED)
        _text(canvas, value, (190, sy), 0.4, TEXT)
        sy += 22

    # ---- event log ------------------------------------------------------
    _panel(canvas, rx, 542, rw, 100, "RECENT EVENTS")
    ey = 576
    for entry in list(meta.get("recent_events", []))[-4:]:
        sev_colour = {"critical": RED, "warning": YELLOW}.get(entry[1], MUTED)
        _text(canvas, entry[0], (rx + 12, ey), 0.38, sev_colour)
        ey += 18

    # ---- footer ---------------------------------------------------------
    _text(canvas, "[q] quit   [c] recalibrate   [r] reset counters   [h] simulate link loss",
          (18, H - 10), 0.38, MUTED)
    if state_obj.reasons:
        _text(canvas, " | ".join(state_obj.reasons[:3]), (rx, H - 10), 0.38, colour)
    return canvas
