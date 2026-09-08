# The detection algorithm

This document maps the pseudocode in Section 8 of the abstract onto the running
code, and argues the four places where the implementation deliberately differs.

All code references are to `edge/fatigue.py` unless stated otherwise.

---

## 1. Mapping to the abstract

| Abstract step | Implementation |
|---|---|
| 1. Initialise camera, sensors, comms | `EdgeNode.__init__` (`edge/node.py`) |
| 2. Capture operator baseline | `Calibrator` |
| 3. Capture frame and sensor readings | `EdgeNode.run` loop head |
| 4. Detect face / not-visible alert | `FatigueEngine.update`, `no_face_frames_alert` |
| 5. Extract facial landmarks | `MediaPipeBackend.process` (`edge/vision.py`) |
| 6. Compute EAR → drowsiness counter | `eye_aspect_ratio`, `_eye_det`, `_c_drowsy` |
| 7. Compute MAR → yawn counter | `mouth_aspect_ratio`, `_yawn_det`, `_c_yawn` |
| 8. Head tilt → posture counter | `tilt_from_accel` (`edge/sensors.py`), `_tilt_det` |
| 9. Heart rate → vitals counter | `HeartRateAnalyser`, `_hr_det` |
| 10. Composite fatigue score | `FatigueEngine.update`, scoring block |
| 11. Classify Normal/Warning/Critical | `_classify` |
| 12–13. Local + remote alerting | `edge/alerts.py`, `edge/telemetry.py` |
| 14. Log to database | `server/database.py::ingest` |
| 15. Reset counters each window | `SlidingCounter` (see §3.1) |
| 16. Shift summary report | `_print_shift_summary`, `/report` page |

---

## 2. The measurements

### Eye Aspect Ratio

```
EAR = (‖p₂ − p₆‖ + ‖p₃ − p₅‖) / (2 ‖p₁ − p₄‖)
```

Six landmarks per eye, from the 478-point MediaPipe FaceMesh. Because the
numerator and denominator are both distances on the same face, EAR is **scale
invariant** — the operator leaning toward or away from the camera does not
change it. A wide-open eye sits near 0.30; a closed eye collapses toward 0.

### Mouth Aspect Ratio

```
MAR = (‖p₂ − p₈‖ + ‖p₃ − p₇‖ + ‖p₄ − p₆‖) / (3 ‖p₁ − p₅‖)
```

Three vertical samples across the inner lip contour rather than one, so a
lopsided or partly covered mouth still measures sensibly. Quiet ≈ 0.05–0.20,
yawn > 0.60.

### PERCLOS

The fraction of a rolling 60-second window spent with the eyes below the
closure threshold. This is the most validated ocular drowsiness measure in the
driver-fatigue literature, and it catches the slow eyelid droop that discrete
microsleep counting misses entirely.

### Head tilt

Pitch and roll from the MPU6050's gravity vector. The IMU is preferred over the
camera for this because **it keeps working when the face leaves the frame** —
which is exactly what happens as the head drops forward.

### Heart-rate variability

RMSSD over a rolling 30-beat window, computed from beat-to-beat intervals.
RMSSD falls as parasympathetic modulation collapses on the way into sleep, so
it carries information the camera cannot see. Intervals outside 300–2000 ms are
rejected as motion artefacts.

---

## 3. The four deliberate departures

### 3.1 Sliding window, not a tumbling reset

> Abstract step 15: *"Reset relevant counters after each evaluation window."*

**The problem.** A tumbling reset means that at the moment a window boundary
passes, every counter drops to zero and the fatigue score collapses with it.
An operator who was Critical at 14:59:59 is Normal at 15:00:00 — not because
anything changed about them, but because a timer expired. The alarm would
silence itself while the hazard was still present.

**What is implemented.** `SlidingCounter` keeps the timestamps of events and
counts how many fall inside the last `window_s` seconds. The counting semantics
are identical to the abstract — "how many microsleeps in the last minute" — but
the answer is continuous in time and no boundary artefact exists. Counters can
still be reset explicitly at a shift change (`reset_window`, and the
supervisor's `reset_counters` command).

### 3.2 Duration gates and refractory periods

**The problem.** The abstract compares each measurement to a threshold once per
frame. At 15 fps a three-second yawn crosses `MAR > MAR_THRESHOLD` on 45
consecutive frames and increments `yawn_counter` 45 times. The score saturates
instantly on a single ordinary yawn. In the other direction, a raw
`EAR < threshold` test fires on every normal blink, and humans blink about
15 times a minute.

**What is implemented.** `GatedDetector` requires a condition to hold for
`min_duration` before it fires at all, then stays silent for `refractory`
seconds. A 250 ms blink never reaches the 400 ms microsleep gate; a yawn counts
once.

The detector distinguishes two kinds of indicator:

- **Discrete acts** (`repeat=False`) — a yawn, a nod. One occurrence is one
  count however long it lasts.
- **Sustained conditions** (`repeat=True`) — eye closure, heart-rate deviation.
  These re-fire every `refractory` seconds for as long as they hold, so an
  operator whose eyes *stay* shut keeps accumulating counts instead of
  registering once and then quietly ageing out of the window.

Logging is separated from counting: only the first fire of an excursion is
written to the event log (`is_new_excursion`), so one long microsleep is one
line in the control room rather than seven, while still scoring higher than a
short one.

### 3.3 Dominant-indicator escalation

> Abstract step 10:
> `Fatigue_Score = w1·drowsiness + w2·yawn + w3·posture + w4·vitals`

**The problem.** Once each counter is normalised to 0–1 so the score is an
interpretable 0–100 index, the weighted mean has a hard ceiling per indicator.
With `w_drowsiness = 0.45`, an operator whose eyes are shut 40 % of the minute —
with the other three channels quiet — scores:

```
100 × (0.45 × 1.0 + 0.20 × 0 + 0.20 × 0 + 0.15 × 0) / 1.00  =  45
```

45 is below the critical threshold of 60. **The system would call a sleeping
operator "Warning" and never escalate**, because the three quiet indicators
dilute the one that matters. This was caught by
`test_sustained_eye_closure_escalates_to_critical`.

**What is implemented.** Each indicator carries an *escalation ceiling* — the
score it is permitted to reach on its own evidence alone — and the final score
is the greater of the weighted mean and the strongest single indicator:

```
weighted = Σ wᵢ·sᵢ / Σ wᵢ
dominant = maxᵢ (sᵢ · escalationᵢ)
score    = 100 × max(weighted, dominant)
```

| Indicator | Ceiling | Reasoning |
|---|---|---|
| Drowsiness | 1.00 | Eyes shut long enough is conclusive by itself. |
| Posture | 0.85 | Repeated nodding off is nearly as conclusive. |
| Vitals | 0.60 | Abnormal vitals alone warrant a warning, not a halt. |
| Yawn | 0.50 | Yawning alone is a symptom, never a verdict. |

The weighted mean still does the work when several indicators are moderately
raised — which is the case the abstract's formula was designed for. The
escalation term only takes over when one indicator is severe and the others are
silent, which is precisely the case the mean handles wrongly.

### 3.4 Hysteresis and dwell

**The problem.** A bare `if score < LOW: Normal elif score < HIGH: Warning`
comparison flips state on every sample while the score hovers on a boundary.
The buzzer would stutter on and off, and the alert feed would fill with
meaningless transitions.

**What is implemented.** `_classify` applies a `hysteresis` margin: leaving a
state requires crossing back past the threshold by that margin. On top of that,
escalation is immediate but **de-escalation must be sustained for `dwell_s`**,
so a momentary dip in the score cannot cancel a live alert. The score itself is
also exponentially smoothed (`smoothing`).

---

## 4. Adaptive calibration

The abstract calls for baseline capture at startup (step 2), and the
implementation uses it for more than a reference point: the eye-closure
threshold is derived from the operator's *own* open-eye EAR:

```
EAR_THRESHOLD = p75(baseline EAR) × 0.72
```

A single global constant such as 0.21 works poorly across a real crew —
absolute EAR varies substantially between individuals, and spectacle wearers
frequently sit below 0.21 with their eyes fully open, which would make them
permanently "asleep" to the system.

### Why the 75th percentile, not the mean or the median

The open-eye state is the **upper mode** of the sample distribution, and the
statistic has to survive contamination from closures.

- A **mean** is dragged down by every blink.
- A **median** tolerates the ~8 % of frames lost to ordinary blinking, but
  collapses onto the *closed* value once closures approach half the samples.
- The **75th percentile** survives that, and still sits within the open-eye
  cluster for an alert operator.

This was not a theoretical concern. During testing, a slow camera-open path
delayed startup by 105 seconds, so the scripted operator had already been
"working" for that long and was microsleeping *through calibration*. The median
landed on 0.100 — a blink — and set the closure threshold to 0.072. Nothing
errored. The system simply ran with a threshold no real eye could ever cross.

### Calibration can refuse

That failure mode is the dangerous one for a safety device: it is **silent**.
The unit reports Normal all shift, the dashboard looks healthy, and nothing is
being detected. So calibration now validates its own result and refuses a
baseline it cannot believe:

| Check | Rejects |
|---|---|
| `plausible_ear_range` (0.15–0.45) | a camera aimed at a half-closed or obscured eye |
| `max_closed_fraction` (0.35) | an operator who was already dozing during calibration |
| `min_valid_samples` | a face that was never properly seen |

On rejection the system falls back to published absolute thresholds, sets
`calibrated = False`, and states the reason — in the cabin log, in the
telemetry record, and on the dashboard, which shows *"Calibration: fallback
thresholds"* instead of *"operator baseline"*. It degrades loudly.

Tests: `test_rejects_an_implausible_baseline`,
`test_rejects_a_baseline_from_an_already_drowsy_operator`,
`test_tolerates_ordinary_blinking_during_calibration`.

---

## 5. Graceful degradation of the score

A signal that is unavailable is **removed from the weighted mean**, not scored
as zero:

```python
available = dict(self.weights)
if not sensors.valid or hr is None:
    available.pop("vitals", None)
...
weighted = Σ(sub[k] · w for k in available) / Σ(w for k in available)
```

Scoring a missing sensor as zero would mean that unplugging the pulse sensor
makes a fatigued operator's score *drop* by up to 15 points. A safety system
must never look safer because it went partly blind. Test:
`test_losing_the_heart_rate_sensor_does_not_deflate_the_score`.

The one indicator that is *not* dropped when it goes missing is the operator
themselves: a face absent for `no_face_frames_alert` consecutive frames raises
`NoOperator` and floors the score at the critical threshold.

---

## 6. Parameter reference

Every constant lives in `config/system_config.yaml` with its reasoning attached.
The values that most affect behaviour:

| Parameter | Default | Effect |
|---|---|---|
| `eye_closed_seconds` | 0.40 s | Closure this long counts as a microsleep. Lower → more sensitive, more false positives from slow blinks. |
| `perclos_warning` / `_critical` | 0.15 / 0.30 | Standard PERCLOS drowsiness bands. |
| `perclos_min_coverage` | 0.50 | PERCLOS is ignored until the window is half full — otherwise the first blink after startup reads as 100 % eye closure. |
| `ear_threshold_ratio` | 0.72 | Fraction of baseline EAR that counts as closed. |
| `plausible_ear_range` | 0.15–0.45 | Calibration refuses a baseline outside this. |
| `max_closed_fraction` | 0.35 | Calibration refuses a baseline captured from a drowsy operator. |
| `scoring.window_s` | 60 s | Evaluation window for all counters. |
| `thresholds.low` / `.high` | 30 / 60 | Warning and Critical boundaries. |
| `hysteresis` | 6.0 | Margin required to leave a state. |
| `dwell_s` | 2.0 | De-escalation must be sustained this long. |

---

## 7. What would come next

The scoring stage is deliberately isolated behind `FatigueEngine.update`, so
the deterministic model can be replaced by a trained classifier without
touching vision, sensors, alerting, telemetry or the control room. The path:

1. Record labelled sessions — the telemetry log already contains every input
   feature the model would need, one row per sample.
2. Train a lightweight classifier (the abstract suggests TensorFlow Lite) on
   those features.
3. Replace the scoring block with inference, keeping the same 0–100 output so
   the state machine, thresholds, alerting and dashboard are unchanged.
4. Validate against held-out operators — not against the same people the
   thresholds were tuned on.

Until step 4 is done, the deterministic model has one decisive advantage for a
safety system: when it raises an alarm, it can say exactly why, and the
dashboard shows that reasoning in plain words.
