"""
Fatigue scoring engine - the implementation of Section 8 of the abstract.

Pipeline per sample:

    raw signals (EAR, MAR, tilt, HR/HRV)
        -> event detection with duration gates and refractory periods
        -> indicator counters over a sliding evaluation window
        -> weighted composite Fatigue Score, 0..100
        -> hysteresis + dwell state machine -> Normal / Warning / Critical

Four deliberate refinements over the abstract pseudocode, all explained in
docs/algorithm.md:

  1. The abstract resets counters at the end of each fixed window (step 15).
     A tumbling reset makes the score collapse to zero the instant a window
     rolls over, which would silence an alert while the operator is still
     impaired. This engine keeps a *sliding* window of event timestamps
     instead: identical counting semantics, but continuous in time.

  2. Raw threshold crossings are gated by a minimum duration and a refractory
     period. Without them a single yawn is counted dozens of times, once per
     frame, and speech or a laugh trips the yawn detector.

  3. The plain weighted sum of step 10 cannot let one indicator raise the
     alarm: with drowsiness weighted 0.45, an operator whose eyes are shut
     40 % of the minute can never score above 45 and is never called Critical.
     A dominant-indicator escalation term fixes that - see _compose_score.

  4. Classification uses hysteresis and a dwell time, so the state cannot
     chatter while the score sits on a threshold, and de-escalation is slower
     than escalation.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

STATE_NORMAL = "Normal"
STATE_WARNING = "Warning"
STATE_CRITICAL = "Critical"
STATE_NO_OPERATOR = "NoOperator"

STATE_SEVERITY = {
    STATE_NORMAL: 0,
    STATE_WARNING: 1,
    STATE_NO_OPERATOR: 2,
    STATE_CRITICAL: 3,
}


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass
class FatigueEvent:
    """A discrete, loggable occurrence worth reporting to the control room."""

    kind: str                 # microsleep | yawn | nod_off | hr_anomaly | ...
    severity: str             # info | warning | critical
    message: str
    value: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------------------
# Calibration - algorithm step 2
# --------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    ear_baseline: Optional[float] = None
    mar_baseline: Optional[float] = None
    hr_baseline: Optional[float] = None
    ear_threshold: float = 0.21
    mar_threshold: float = 0.60
    samples: int = 0
    ok: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ear_baseline": self.ear_baseline,
            "mar_baseline": self.mar_baseline,
            "hr_baseline": self.hr_baseline,
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "samples": self.samples,
            "ok": self.ok,
            "note": self.note,
        }


class Calibrator:
    """
    Captures the operator's own baseline during the first seconds of a shift.

    Deriving the eye-closure threshold from the operator's own open-eye EAR
    (rather than using one global constant) is what lets the system work for
    people with different eye shapes and for spectacle wearers, whose absolute
    EAR can sit well below the textbook 0.21.
    """

    def __init__(self, cfg):
        cal = cfg.calibration
        self.duration = float(cal.duration_s)
        self.min_samples = int(cal.min_valid_samples)
        self.ear_ratio = float(cal.ear_threshold_ratio)
        self.mar_ratio = float(cal.mar_threshold_ratio)
        self.fallback_ear = float(cal.fallback_ear_threshold)
        self.fallback_mar = float(cal.fallback_mar_threshold)
        plausible = cal.get("plausible_ear_range", [0.15, 0.45])
        self.plausible_min = float(plausible[0])
        self.plausible_max = float(plausible[1])
        self.max_closed_fraction = float(cal.get("max_closed_fraction", 0.35))

        self._ear: list = []
        self._mar: list = []
        self._hr: list = []
        self._started: Optional[float] = None
        self.result: Optional[CalibrationResult] = None

    @property
    def in_progress(self) -> bool:
        return self.result is None

    def progress(self) -> float:
        if self._started is None:
            return 0.0
        return min(1.0, (time.time() - self._started) / self.duration)

    def feed(self, face, sensors, now: Optional[float] = None) -> Optional[CalibrationResult]:
        """Accumulate a sample; returns the result once calibration finishes."""
        now = time.time() if now is None else now
        if self.result is not None:
            return self.result
        if self._started is None:
            self._started = now

        if face.face_found and face.ear is not None and face.quality > 0.3:
            self._ear.append(face.ear)
            if face.mar is not None:
                self._mar.append(face.mar)
        if sensors.valid and sensors.heart_rate:
            self._hr.append(sensors.heart_rate)

        if now - self._started < self.duration:
            return None

        self.result = self._finalise()
        return self.result

    def _finalise(self) -> CalibrationResult:
        import statistics

        result = CalibrationResult(samples=len(self._ear))

        if len(self._ear) >= self.min_samples:
            ordered = sorted(self._ear)
            # The open-eye state is the upper mode of the distribution, so the
            # baseline is taken at the 75th percentile rather than the median.
            # The median tolerates the ~8 % of frames lost to ordinary blinks,
            # but collapses onto the closed-eye value once closures approach
            # half the samples; the upper quartile survives that.
            baseline = float(ordered[min(len(ordered) - 1,
                                         int(0.75 * len(ordered)))])
            closed = sum(1 for e in self._ear if e < baseline * self.ear_ratio)
            closed_fraction = closed / float(len(self._ear))

            if not (self.plausible_min <= baseline <= self.plausible_max):
                # Not a credible open eye - refuse it rather than desensitise
                # the detector for the rest of the shift.
                result.ear_threshold = self.fallback_ear
                result.note = ("baseline EAR %.3f is outside the plausible "
                               "range %.2f-%.2f, using fallback thresholds"
                               % (baseline, self.plausible_min, self.plausible_max))
            elif closed_fraction > self.max_closed_fraction:
                result.ear_threshold = self.fallback_ear
                result.note = ("operator's eyes were closed for %.0f%% of "
                               "calibration - not a valid alert baseline, "
                               "using fallback thresholds" % (closed_fraction * 100))
            else:
                result.ear_baseline = baseline
                result.ear_threshold = baseline * self.ear_ratio
                result.ok = True
                result.note = "calibrated from %d frames" % len(self._ear)
        else:
            result.ear_threshold = self.fallback_ear
            result.note = ("only %d valid frames, using fallback thresholds"
                           % len(self._ear))

        if len(self._mar) >= self.min_samples:
            result.mar_baseline = float(statistics.median(self._mar))
            result.mar_threshold = max(result.mar_baseline * self.mar_ratio,
                                       self.fallback_mar * 0.6)
        else:
            result.mar_threshold = self.fallback_mar

        if len(self._hr) >= 10:
            result.hr_baseline = float(statistics.median(self._hr))

        return result

    def force_default(self, note: str = "calibration skipped") -> CalibrationResult:
        self.result = CalibrationResult(ear_threshold=self.fallback_ear,
                                        mar_threshold=self.fallback_mar,
                                        ok=False, note=note)
        return self.result


# --------------------------------------------------------------------------
# Duration-gated threshold detector
# --------------------------------------------------------------------------


class GatedDetector:
    """
    Fires once per genuine excursion past a threshold.

    A raw comparison fires on every frame, so one three-second yawn would be
    counted 45 times. This detector requires the condition to hold for
    `min_duration` before firing, then stays quiet for `refractory` seconds.

    `repeat` selects between the two kinds of indicator:

      repeat=False (yawn, nod-off) - discrete acts. One yawn is one yawn,
        however long it lasts, and the counter must not be inflated by it.

      repeat=True (eye closure, heart-rate anomaly) - sustained conditions.
        These re-fire every `refractory` seconds for as long as they hold, so
        that an operator whose eyes stay shut, or whose heart rate stays
        depressed, keeps accumulating counts instead of registering once and
        then quietly ageing out of the evaluation window.
    """

    def __init__(self, min_duration: float, refractory: float, repeat: bool = False):
        self.min_duration = float(min_duration)
        self.refractory = float(refractory)
        self.repeat = bool(repeat)
        self._active_since: Optional[float] = None
        self._last_fire: float = -1e9
        self._fired_this_excursion = False
        # True when the most recent fire was the *start* of an excursion rather
        # than a repeat within one. The counter wants every fire, because it is
        # measuring how long the condition held; the event log wants only the
        # first, or one three-second microsleep becomes seven log entries.
        self.is_new_excursion = False

    def update(self, condition: bool, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        if not condition:
            self._active_since = None
            self._fired_this_excursion = False
            return False

        if self._active_since is None:
            self._active_since = now
            return False

        if now - self._active_since < self.min_duration:
            return False
        if now - self._last_fire < self.refractory:
            return False
        if self._fired_this_excursion and not self.repeat:
            return False

        self.is_new_excursion = not self._fired_this_excursion
        self._last_fire = now
        self._fired_this_excursion = True
        return True

    @property
    def held_for(self) -> float:
        if self._active_since is None:
            return 0.0
        return time.time() - self._active_since


class SlidingCounter:
    """Counts events that occurred within the last `window` seconds."""

    def __init__(self, window: float):
        self.window = float(window)
        self._stamps: deque = deque()

    def add(self, timestamp: Optional[float] = None) -> None:
        self._stamps.append(timestamp if timestamp is not None else time.time())

    def count(self, now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        cutoff = now - self.window
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.popleft()
        return len(self._stamps)

    def clear(self) -> None:
        self._stamps.clear()


class PerclosTracker:
    """
    PERCLOS - the fraction of time the eyes are closed over a rolling window.

    This is the single most validated ocular fatigue measure in the driver
    drowsiness literature, and it captures slow droop that discrete microsleep
    counting misses entirely.
    """

    def __init__(self, window: float):
        self.window = float(window)
        self._samples: deque = deque()      # (timestamp, dt, closed)

    def add(self, closed: bool, dt: float, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        self._samples.append((now, max(0.0, min(dt, 1.0)), bool(closed)))
        cutoff = now - self.window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def value(self) -> float:
        total = sum(s[1] for s in self._samples)
        if total <= 1e-6:
            return 0.0
        closed = sum(s[1] for s in self._samples if s[2])
        return closed / total

    def coverage(self) -> float:
        """Fraction of the window actually filled with data (0..1)."""
        total = sum(s[1] for s in self._samples)
        return min(1.0, total / self.window) if self.window > 0 else 0.0

    def clear(self) -> None:
        self._samples.clear()


# --------------------------------------------------------------------------
# Engine output
# --------------------------------------------------------------------------


@dataclass
class FatigueState:
    """The complete evaluated state for one sample - what gets published."""

    timestamp: float = field(default_factory=time.time)
    state: str = STATE_NORMAL
    previous_state: str = STATE_NORMAL
    state_changed: bool = False
    score: float = 0.0                # smoothed, 0..100
    raw_score: float = 0.0
    sub_scores: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    perclos: float = 0.0
    face_visible: bool = False
    ear: Optional[float] = None
    mar: Optional[float] = None
    tilt: Optional[float] = None
    heart_rate: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    ear_threshold: float = 0.0
    mar_threshold: float = 0.0
    reasons: list = field(default_factory=list)
    events: list = field(default_factory=list)
    vision_backend: str = "none"
    sensor_source: str = "none"
    calibrated: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state": self.state,
            "previous_state": self.previous_state,
            "state_changed": self.state_changed,
            "score": round(self.score, 2),
            "raw_score": round(self.raw_score, 2),
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "counters": self.counters,
            "perclos": round(self.perclos, 4),
            "face_visible": self.face_visible,
            "ear": round(self.ear, 4) if self.ear is not None else None,
            "mar": round(self.mar, 4) if self.mar is not None else None,
            "tilt": round(self.tilt, 2) if self.tilt is not None else None,
            "heart_rate": round(self.heart_rate, 1) if self.heart_rate else None,
            "hrv_rmssd": round(self.hrv_rmssd, 1) if self.hrv_rmssd else None,
            "ear_threshold": round(self.ear_threshold, 4),
            "mar_threshold": round(self.mar_threshold, 4),
            "reasons": self.reasons,
            "events": [e.to_dict() for e in self.events],
            "vision_backend": self.vision_backend,
            "sensor_source": self.sensor_source,
            "calibrated": self.calibrated,
        }


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class FatigueEngine:
    """Turns per-frame measurements into a fatigue state. Pure logic, no I/O."""

    def __init__(self, cfg):
        self.cfg = cfg
        det = cfg.detection
        sc = cfg.scoring

        self.window_s = float(sc.window_s)
        self.weights = dict(sc.weights)
        self.saturation = dict(sc.saturation)
        self.escalation = dict(sc.get("escalation", {}))
        self.low_threshold = float(sc.thresholds.low)
        self.high_threshold = float(sc.thresholds.high)
        self.hysteresis = float(sc.hysteresis)
        self.dwell_s = float(sc.dwell_s)
        self.smoothing = float(sc.smoothing)

        self.eye_closed_seconds = float(det.eye_closed_seconds)
        self.no_face_frames_alert = int(det.no_face_frames_alert)
        self.perclos_warning = float(det.perclos_warning)
        self.perclos_critical = float(det.perclos_critical)
        self.perclos_min_coverage = float(det.get("perclos_min_coverage", 0.5))
        self.tilt_threshold = float(det.tilt_threshold_deg)
        self.hr_deviation = float(det.hr_deviation_bpm)
        self.hr_low_absolute = float(det.hr_low_absolute)

        # Duration-gated detectors, one per indicator family. Eye closure and
        # heart-rate deviation are sustained conditions and so re-fire while
        # they hold; a yawn and a nod are single acts and fire once each.
        self._eye_det = GatedDetector(det.eye_closed_seconds,
                                      det.eye_closed_seconds, repeat=True)
        self._yawn_det = GatedDetector(det.yawn_min_duration_s, det.yawn_refractory_s)
        self._tilt_det = GatedDetector(det.tilt_min_duration_s, det.tilt_refractory_s)
        self._hr_det = GatedDetector(det.hr_min_duration_s, det.hr_refractory_s,
                                     repeat=True)

        # Sliding counters over the evaluation window.
        self._c_drowsy = SlidingCounter(self.window_s)
        self._c_yawn = SlidingCounter(self.window_s)
        self._c_posture = SlidingCounter(self.window_s)
        self._c_vitals = SlidingCounter(self.window_s)
        self._perclos = PerclosTracker(float(det.perclos_window_s))

        self.calibration = CalibrationResult()
        self._score = 0.0
        self._state = STATE_NORMAL
        self._candidate_state = STATE_NORMAL
        self._candidate_since = time.time()
        self._no_face_frames = 0
        self._no_face_alerted = False
        self._last_update: Optional[float] = None

    # -- setup ------------------------------------------------------------

    def apply_calibration(self, calibration: CalibrationResult) -> None:
        self.calibration = calibration

    def reset_window(self) -> None:
        """Explicit counter reset (algorithm step 15), e.g. at shift change."""
        for counter in (self._c_drowsy, self._c_yawn, self._c_posture, self._c_vitals):
            counter.clear()
        self._perclos.clear()

    # -- main entry point -------------------------------------------------

    def update(self, face, sensors, now: Optional[float] = None) -> FatigueState:
        """Process one synchronised (frame, sensor) pair. Steps 3-13."""
        now = now if now is not None else time.time()
        dt = 0.0 if self._last_update is None else max(0.0, now - self._last_update)
        self._last_update = now

        events: list = []
        reasons: list = []

        ear_threshold = self.calibration.ear_threshold
        mar_threshold = self.calibration.mar_threshold

        # --- step 4: operator visible? -----------------------------------
        if face.face_found:
            if self._no_face_alerted:
                events.append(FatigueEvent(
                    "operator_returned", "info", "Operator back in frame"))
            self._no_face_frames = 0
            self._no_face_alerted = False
        else:
            self._no_face_frames += 1
            if (self._no_face_frames >= self.no_face_frames_alert
                    and not self._no_face_alerted):
                self._no_face_alerted = True
                events.append(FatigueEvent(
                    "operator_not_visible", "critical",
                    "Operator not visible for %d consecutive frames"
                    % self._no_face_frames, float(self._no_face_frames)))

        # --- step 6: eye closure -> drowsiness ----------------------------
        eyes_closed = bool(face.face_found and face.ear is not None
                           and face.ear < ear_threshold)
        if face.face_found:
            self._perclos.add(eyes_closed, dt, now)
        if self._eye_det.update(eyes_closed, now):
            # Every fire counts toward the drowsiness counter - a longer closure
            # is genuinely more dangerous - but only the first is logged, so one
            # microsleep is one line in the control-room feed.
            self._c_drowsy.add(now)
            if self._eye_det.is_new_excursion:
                events.append(FatigueEvent(
                    "microsleep", "warning",
                    "Eyes closed past %.1fs (EAR %.3f < %.3f)"
                    % (self.eye_closed_seconds, face.ear or 0.0, ear_threshold),
                    face.ear))

        # --- step 7: yawning ----------------------------------------------
        yawning = bool(face.face_found and face.mar is not None
                       and face.mar > mar_threshold)
        if self._yawn_det.update(yawning, now):
            self._c_yawn.add(now)
            events.append(FatigueEvent(
                "yawn", "info",
                "Yawn detected (MAR %.3f > %.3f)" % (face.mar or 0.0, mar_threshold),
                face.mar))

        # --- step 8: head tilt / nodding off ------------------------------
        # Prefer the IMU: it keeps working when the face leaves the frame,
        # which is exactly what happens as the head drops.
        if sensors.valid:
            tilt = max(abs(sensors.tilt_pitch), abs(sensors.tilt_roll))
        elif face.face_found and face.pitch is not None:
            tilt = max(abs(face.pitch), abs(face.roll or 0.0))
        else:
            tilt = None

        if tilt is not None and self._tilt_det.update(tilt > self.tilt_threshold, now):
            self._c_posture.add(now)
            events.append(FatigueEvent(
                "nod_off", "warning",
                "Nodding-off posture: head tilt %.1f deg" % tilt, tilt))

        # --- step 9: heart-rate deviation ---------------------------------
        hr = sensors.heart_rate if sensors.valid else None
        hr_baseline = self.calibration.hr_baseline
        hr_abnormal = False
        if hr is not None:
            if hr < self.hr_low_absolute:
                hr_abnormal = True
            elif hr_baseline and abs(hr - hr_baseline) > self.hr_deviation:
                hr_abnormal = True
        if hr is not None and self._hr_det.update(hr_abnormal, now):
            self._c_vitals.add(now)
            if self._hr_det.is_new_excursion:
                base_txt = ("baseline %.0f" % hr_baseline) if hr_baseline else "no baseline"
                events.append(FatigueEvent(
                    "hr_anomaly", "warning",
                    "Heart rate %.0f bpm deviates from %s" % (hr, base_txt), hr))

        # --- step 10: composite fatigue score -----------------------------
        counters = {
            "drowsiness": self._c_drowsy.count(now),
            "yawn": self._c_yawn.count(now),
            "posture": self._c_posture.count(now),
            "vitals": self._c_vitals.count(now),
        }
        perclos = self._perclos.value()

        sub_scores = {
            "drowsiness": self._drowsiness_sub_score(counters["drowsiness"], perclos,
                                                     self._perclos.coverage()),
            "yawn": self._normalise(counters["yawn"], "yawn"),
            "posture": self._normalise(counters["posture"], "posture"),
            "vitals": self._vitals_sub_score(counters["vitals"], sensors),
        }

        # Signals that are unavailable are dropped from the weighted mean
        # rather than counted as zero, otherwise losing the heart-rate sensor
        # would make a tired operator look 15 % fresher.
        available = dict(self.weights)
        if not sensors.valid or hr is None:
            available.pop("vitals", None)
        if not face.face_found:
            available.pop("yawn", None)
            if self._perclos.coverage() < 0.1:
                available.pop("drowsiness", None)

        weight_sum = sum(available.values())
        if weight_sum <= 0:
            raw_score = 0.0
        else:
            # The abstract's composite: a weighted mean of the four indicators.
            weighted = sum(sub_scores[k] * w for k, w in available.items()) / weight_sum

            # Dominant-indicator escalation. A weighted mean cannot express
            # "this one signal is sufficient on its own", yet an operator whose
            # eyes are shut is impaired no matter how calm the other three
            # channels look. Each indicator therefore also carries a ceiling it
            # may drive the score to unaided, and the higher of the two wins.
            dominant = max((sub_scores[k] * self.escalation.get(k, 0.5)
                            for k in available), default=0.0)

            raw_score = 100.0 * max(weighted, dominant)

        # An operator who has vanished from frame is a safety event in its own
        # right, so the score floors at the warning threshold while that holds.
        if self._no_face_alerted:
            raw_score = max(raw_score, self.high_threshold)

        raw_score = float(max(0.0, min(100.0, raw_score)))
        alpha = max(0.0, min(0.95, self.smoothing))
        self._score = alpha * self._score + (1.0 - alpha) * raw_score

        # --- step 11: classify with hysteresis and dwell -------------------
        target = self._classify(self._score)
        if self._no_face_alerted:
            target = STATE_NO_OPERATOR

        if target != self._candidate_state:
            self._candidate_state = target
            self._candidate_since = now

        previous_state = self._state
        state_changed = False
        escalating = STATE_SEVERITY[self._candidate_state] > STATE_SEVERITY[self._state]
        # Escalate immediately; de-escalate only after the dwell time, so a
        # momentary dip in the score cannot cancel a live alert.
        if self._candidate_state != self._state:
            if escalating or (now - self._candidate_since) >= self.dwell_s:
                self._state = self._candidate_state
                state_changed = True
                events.append(FatigueEvent(
                    "state_change",
                    self._severity_for(self._state),
                    "%s -> %s (score %.1f)" % (previous_state, self._state, self._score),
                    self._score))

        # --- human-readable explanation of the current score ---------------
        if counters["drowsiness"]:
            reasons.append("%d microsleep event(s) in the last %.0fs"
                           % (counters["drowsiness"], self.window_s))
        if (perclos > self.perclos_warning
                and self._perclos.coverage() >= self.perclos_min_coverage):
            reasons.append("PERCLOS %.0f%% of the last minute" % (perclos * 100))
        if counters["yawn"]:
            reasons.append("%d yawn(s)" % counters["yawn"])
        if counters["posture"]:
            reasons.append("%d nodding-off posture event(s)" % counters["posture"])
        if counters["vitals"]:
            reasons.append("heart-rate anomaly")
        if self._no_face_alerted:
            reasons.append("operator not visible")

        return FatigueState(
            timestamp=now,
            state=self._state, previous_state=previous_state,
            state_changed=state_changed,
            score=self._score, raw_score=raw_score,
            sub_scores=sub_scores, counters=counters, perclos=perclos,
            face_visible=bool(face.face_found),
            ear=face.ear, mar=face.mar, tilt=tilt,
            heart_rate=hr, hrv_rmssd=sensors.hrv_rmssd if sensors.valid else None,
            ear_threshold=ear_threshold, mar_threshold=mar_threshold,
            reasons=reasons, events=events,
            vision_backend=face.backend, sensor_source=sensors.source,
            calibrated=self.calibration.ok,
        )

    # -- helpers ----------------------------------------------------------

    def _normalise(self, count: int, key: str) -> float:
        saturation = max(1, int(self.saturation.get(key, 3)))
        return min(1.0, count / float(saturation))

    def _drowsiness_sub_score(self, count: int, perclos: float,
                              coverage: float) -> float:
        """
        Drowsiness combines discrete microsleeps with continuous PERCLOS.

        The worse of the two wins: a slow, steady droop and a sequence of sharp
        microsleeps are both dangerous, and either alone should raise the score.

        PERCLOS is only consulted once the rolling window is at least
        `perclos_min_coverage` full. Before that the ratio is computed over a
        second or two of video, where a single 300 ms blink reads as 20 % eye
        closure and would raise a Warning the moment calibration ended.
        """
        from_events = self._normalise(count, "drowsiness")
        if coverage < self.perclos_min_coverage:
            return from_events

        # Piecewise map: 0 at no closure, 0.5 at the warning PERCLOS,
        # 1.0 at the critical PERCLOS.
        if perclos <= self.perclos_warning:
            from_perclos = 0.5 * perclos / max(1e-6, self.perclos_warning)
        else:
            span = max(1e-6, self.perclos_critical - self.perclos_warning)
            from_perclos = 0.5 + 0.5 * (perclos - self.perclos_warning) / span
        from_perclos = max(0.0, min(1.0, from_perclos))

        return max(from_events, from_perclos)

    def _vitals_sub_score(self, count: int, sensors) -> float:
        """Counter-driven, with a bonus for collapsed heart-rate variability."""
        score = self._normalise(count, "vitals")
        if sensors.valid and sensors.hrv_rmssd is not None:
            # RMSSD below ~20 ms in a seated adult indicates strongly reduced
            # parasympathetic modulation, a recognised drowsiness correlate.
            if sensors.hrv_rmssd < 20.0:
                score = max(score, min(1.0, (20.0 - sensors.hrv_rmssd) / 15.0))
        return score

    def _classify(self, score: float) -> str:
        """Step 11, with hysteresis so the state cannot chatter on a boundary."""
        if self._state == STATE_NORMAL:
            if score >= self.high_threshold:
                return STATE_CRITICAL
            if score >= self.low_threshold:
                return STATE_WARNING
            return STATE_NORMAL
        if self._state == STATE_WARNING:
            if score >= self.high_threshold:
                return STATE_CRITICAL
            if score < self.low_threshold - self.hysteresis:
                return STATE_NORMAL
            return STATE_WARNING
        # Critical or NoOperator
        if score < self.high_threshold - self.hysteresis:
            return STATE_WARNING if score >= self.low_threshold else STATE_NORMAL
        return STATE_CRITICAL

    @staticmethod
    def _severity_for(state: str) -> str:
        if state in (STATE_CRITICAL, STATE_NO_OPERATOR):
            return "critical"
        if state == STATE_WARNING:
            return "warning"
        return "info"
