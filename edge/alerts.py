"""
Local alert unit - algorithm steps 12 and 13.

Drives the in-cabin annunciators:

    LED green  / yellow / red   visual status
    buzzer                      audible escalation
    vibration motor             haptic, for a noisy cabin where a buzzer is
                                easily missed

Two drivers, selected automatically:

    GpioAlertDriver     real LEDs and buzzer on Raspberry Pi GPIO pins
    DesktopAlertDriver  console status line plus the PC sound card, so the
                        alert logic can be demonstrated without hardware

Both implement the same three-state interface, so nothing above this layer
knows or cares which one is running.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from .fatigue import STATE_CRITICAL, STATE_NO_OPERATOR, STATE_NORMAL, STATE_WARNING


class _PatternPlayer:
    """Runs an on/off beep pattern on a background thread until told to stop."""

    def __init__(self, beep_fn, off_fn):
        self._beep = beep_fn
        self._off = off_fn
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pattern = None
        self._lock = threading.Lock()

    def set_pattern(self, pattern, volume: float = 1.0) -> None:
        """pattern is (on_ms, off_ms) or None to silence."""
        with self._lock:
            if pattern == self._pattern:
                return                    # already playing this pattern
            self._pattern = pattern
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._off()
        if pattern is None:
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(pattern, volume),
                                        daemon=True, name="alert-pattern")
        self._thread.start()

    def _run(self, pattern, volume: float) -> None:
        on_ms, off_ms = pattern
        while not self._stop.is_set():
            self._beep(on_ms, volume)
            if self._stop.wait(off_ms / 1000.0):
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._off()


class DesktopAlertDriver:
    """
    Bench-top stand-in for the cabin alert unit.

    LED colours become an ANSI-coloured status line; the buzzer becomes a tone
    on the PC sound card (winsound on Windows, terminal bell elsewhere).
    """

    name = "desktop"

    def __init__(self, cfg):
        self.cfg = cfg
        self._state = STATE_NORMAL
        self._audio = self._pick_audio()
        self._player = _PatternPlayer(self._audio, lambda: None)

    @staticmethod
    def _pick_audio():
        if sys.platform.startswith("win"):
            try:
                import winsound

                def beep(duration_ms: int, volume: float) -> None:
                    # Frequency carries the urgency; winsound has no volume
                    # control, so a higher pitch stands in for a louder tone.
                    freq = 1500 if volume > 0.6 else 880
                    try:
                        winsound.Beep(freq, max(40, int(duration_ms)))
                    except Exception:
                        pass

                return beep
            except Exception:
                pass

        def beep(duration_ms: int, volume: float) -> None:
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(max(0.04, duration_ms / 1000.0))

        return beep

    def set_state(self, state: str, score: float = 0.0) -> None:
        if state == self._state:
            return
        self._state = state

        if state == STATE_WARNING:
            cfg = self.cfg.alerts.warning
            self._player.set_pattern(tuple(cfg.beep_pattern_ms), float(cfg.volume))
        elif state in (STATE_CRITICAL, STATE_NO_OPERATOR):
            cfg = self.cfg.alerts.critical
            self._player.set_pattern(tuple(cfg.beep_pattern_ms), float(cfg.volume))
        else:
            self._player.set_pattern(None)

    def status_line(self) -> str:
        colours = {STATE_NORMAL: "\033[92m", STATE_WARNING: "\033[93m",
                   STATE_CRITICAL: "\033[91m", STATE_NO_OPERATOR: "\033[95m"}
        leds = {STATE_NORMAL: "GREEN", STATE_WARNING: "YELLOW",
                STATE_CRITICAL: "RED", STATE_NO_OPERATOR: "RED/BLINK"}
        colour = colours.get(self._state, "")
        return "%sLED=%s%s" % (colour, leds.get(self._state, "?"), "\033[0m")

    def close(self) -> None:
        self._player.stop()


class GpioAlertDriver:
    """Real LEDs, buzzer and vibration motor on Raspberry Pi GPIO."""

    name = "gpio"

    def __init__(self, cfg):
        import RPi.GPIO as GPIO           # imported lazily

        self._GPIO = GPIO
        self.cfg = cfg
        pins = cfg.alerts.gpio_pins
        self._pins = {k: int(v) for k, v in pins.items()}

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in self._pins.values():
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        self._buzzer_pwm = GPIO.PWM(self._pins["buzzer"], 2000)
        self._state = STATE_NORMAL
        self._player = _PatternPlayer(self._beep, self._buzzer_off)
        self.set_state(STATE_NORMAL)

    @staticmethod
    def available() -> bool:
        try:
            import RPi.GPIO  # noqa: F401
            return True
        except Exception:
            return False

    def _beep(self, duration_ms: int, volume: float) -> None:
        # Duty cycle sets the perceived loudness of a piezo buzzer.
        self._buzzer_pwm.start(max(5.0, min(50.0, volume * 50.0)))
        time.sleep(max(0.02, duration_ms / 1000.0))
        self._buzzer_pwm.stop()

    def _buzzer_off(self) -> None:
        try:
            self._buzzer_pwm.stop()
        except Exception:
            pass

    def _leds(self, green: bool, yellow: bool, red: bool) -> None:
        self._GPIO.output(self._pins["led_green"], green)
        self._GPIO.output(self._pins["led_yellow"], yellow)
        self._GPIO.output(self._pins["led_red"], red)

    def set_state(self, state: str, score: float = 0.0) -> None:
        if state == self._state:
            return
        self._state = state

        if state == STATE_NORMAL:
            self._leds(True, False, False)
            self._player.set_pattern(None)
            self._GPIO.output(self._pins["vibration"], False)
        elif state == STATE_WARNING:
            self._leds(False, True, False)
            cfg = self.cfg.alerts.warning
            self._player.set_pattern(tuple(cfg.beep_pattern_ms), float(cfg.volume))
            self._GPIO.output(self._pins["vibration"], False)
        else:                                     # Critical or NoOperator
            self._leds(False, False, True)
            cfg = self.cfg.alerts.critical
            self._player.set_pattern(tuple(cfg.beep_pattern_ms), float(cfg.volume))
            self._GPIO.output(self._pins["vibration"], bool(cfg.get("vibration", True)))

    def status_line(self) -> str:
        leds = {STATE_NORMAL: "GREEN", STATE_WARNING: "YELLOW",
                STATE_CRITICAL: "RED", STATE_NO_OPERATOR: "RED/BLINK"}
        return "LED=%s" % leds.get(self._state, "?")

    def close(self) -> None:
        self._player.stop()
        try:
            self._leds(False, False, False)
            self._GPIO.output(self._pins["vibration"], False)
            self._GPIO.cleanup()
        except Exception:
            pass


class NullAlertDriver:
    """No annunciation at all - used by the automated tests."""

    name = "none"

    def __init__(self, cfg=None):
        self._state = STATE_NORMAL

    def set_state(self, state: str, score: float = 0.0) -> None:
        self._state = state

    def status_line(self) -> str:
        return "LED=off"

    def close(self) -> None:
        pass


def create_alert_unit(cfg, logger=None):
    """Build the alert driver named in the config ('auto' prefers real GPIO)."""
    driver = str(cfg.alerts.get("driver", "auto")).lower()

    def _log(message: str) -> None:
        if logger:
            logger(message)

    if driver == "none":
        return NullAlertDriver(cfg)
    if driver == "desktop":
        return DesktopAlertDriver(cfg)
    if driver == "gpio":
        return GpioAlertDriver(cfg)

    if GpioAlertDriver.available():
        try:
            unit = GpioAlertDriver(cfg)
            _log("alert unit: GPIO (LEDs + buzzer + vibration motor)")
            return unit
        except Exception as exc:
            _log("alert unit: GPIO init failed (%s)" % exc)

    _log("alert unit: desktop simulation (no GPIO available)")
    return DesktopAlertDriver(cfg)
