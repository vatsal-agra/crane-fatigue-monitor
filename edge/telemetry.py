"""
Telemetry link to the control room (the Wi-Fi / GSM leg of the block diagram).

Design constraints that come from the deployment, not from the code:

  * A crane cabin moves through steel structures. The link *will* drop.
    Losing fatigue history because the radio was shadowed for ninety seconds
    is unacceptable, so every sample is written to a disk-backed queue and
    replayed once the link returns (store-and-forward).

  * Publishing must never stall the detection loop. All network I/O happens on
    a worker thread; the detection loop only ever does a non-blocking enqueue.

  * The same client also polls for supervisor commands, which is what closes
    the loop back to the cabin: acknowledge an alert, request a rotation, or
    halt crane operation.

Two transports are provided. HTTP is the default because it needs no broker
and therefore no setup during a demo; MQTT is available for a deployment that
already has a broker.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Optional

import requests


class HttpTransport:
    """Batched HTTP POST to the Flask ingest endpoint."""

    name = "http"

    def __init__(self, cfg):
        base = str(cfg.telemetry.server_url).rstrip("/")
        self.ingest_url = base + str(cfg.telemetry.ingest_path)
        self.command_url = base + str(cfg.telemetry.command_path)
        self.timeout = float(cfg.telemetry.request_timeout_s)
        self._session = requests.Session()

    def send(self, batch: list) -> bool:
        response = self._session.post(self.ingest_url, json={"records": batch},
                                      timeout=self.timeout)
        response.raise_for_status()
        return True

    def fetch_commands(self, device_id: str, since: float) -> list:
        response = self._session.get(self.command_url,
                                     params={"device_id": device_id, "since": since},
                                     timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("commands", [])

    def close(self) -> None:
        self._session.close()


class MqttTransport:
    """Publishes to an MQTT broker; requires `pip install paho-mqtt`."""

    name = "mqtt"

    def __init__(self, cfg):
        import paho.mqtt.client as mqtt         # imported lazily

        mq = cfg.telemetry.mqtt
        self.topic = str(mq.topic_telemetry)
        self.command_topic = str(mq.get("topic_commands", "crane/fatigue/commands"))
        self._commands: list = []
        self._lock = threading.Lock()

        self._client = mqtt.Client()
        self._client.on_message = self._on_message
        self._client.connect(str(mq.host), int(mq.port), keepalive=30)
        self._client.subscribe(self.command_topic)
        self._client.loop_start()

    def _on_message(self, _client, _userdata, message) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except Exception:
            return
        with self._lock:
            self._commands.append(payload)

    def send(self, batch: list) -> bool:
        info = self._client.publish(self.topic, json.dumps({"records": batch}), qos=1)
        info.wait_for_publish(timeout=5.0)
        return info.is_published()

    def fetch_commands(self, device_id: str, since: float) -> list:
        with self._lock:
            pending, self._commands = self._commands, []
        return [c for c in pending if c.get("device_id") in (None, device_id)]

    def close(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass


class TelemetryClient:
    """
    Non-blocking, store-and-forward publisher plus command poller.

    Call publish() from the detection loop as often as you like; it returns
    immediately. Call poll_commands() to collect anything the supervisor has
    sent down.
    """

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.device_id = str(cfg.system.device_id)
        self.batch_max = int(cfg.telemetry.batch_max)
        self.interval = float(cfg.telemetry.publish_interval_s)
        self.buffer_path = str(cfg.telemetry.offline_buffer_path)
        self.buffer_max = int(cfg.telemetry.offline_buffer_max_records)
        self._logger = logger

        self._queue: queue.Queue = queue.Queue(maxsize=5000)
        self._commands: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_command_poll = 0.0
        self._command_cursor = time.time()

        self.connected = False
        self.sent_count = 0
        self.buffered_count = 0
        self.last_error: Optional[str] = None

        transport_name = str(cfg.telemetry.get("transport", "http")).lower()
        self.transport = None
        if transport_name == "http":
            self.transport = HttpTransport(cfg)
        elif transport_name == "mqtt":
            try:
                self.transport = MqttTransport(cfg)
            except Exception as exc:
                self._log("MQTT unavailable (%s); falling back to HTTP" % exc)
                self.transport = HttpTransport(cfg)
        elif transport_name != "none":
            raise ValueError("Unknown telemetry transport %r" % transport_name)

        os.makedirs(os.path.dirname(self.buffer_path) or ".", exist_ok=True)

    def _log(self, message: str) -> None:
        if self._logger:
            self._logger(message)

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        if self.transport is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="telemetry")
        self._thread.start()

    def publish(self, record: dict) -> None:
        """Enqueue a record. Never blocks; drops the oldest if the queue fills."""
        if self.transport is None:
            return
        record.setdefault("device_id", self.device_id)
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            try:
                self._queue.get_nowait()          # drop the oldest
                self._queue.put_nowait(record)
            except queue.Empty:
                pass

    def poll_commands(self) -> list:
        """Return any supervisor commands received since the last call."""
        out = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self.transport:
            self.transport.close()

    def stats(self) -> dict:
        return {
            "transport": self.transport.name if self.transport else "none",
            "connected": self.connected,
            "sent": self.sent_count,
            "buffered": self.buffered_count,
            "queued": self._queue.qsize(),
            "last_error": self.last_error,
        }

    # -- worker ------------------------------------------------------------

    def _worker(self) -> None:
        while self._running:
            batch = self._drain(self.batch_max)
            if batch:
                if self._try_send(batch):
                    self.sent_count += len(batch)
                    self._flush_offline_buffer()
                else:
                    self._write_offline(batch)
            elif self.connected:
                # Idle link is a good moment to retry anything left on disk.
                self._flush_offline_buffer()

            self._maybe_poll_commands()
            time.sleep(self.interval)

        # Drain whatever is left so a clean shutdown loses nothing.
        remaining = self._drain(self.batch_max * 4)
        if remaining and not self._try_send(remaining):
            self._write_offline(remaining)

    def _drain(self, limit: int) -> list:
        batch = []
        while len(batch) < limit:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _try_send(self, batch: list) -> bool:
        try:
            self.transport.send(batch)
            if not self.connected:
                self._log("telemetry link up (%s)" % self.transport.name)
            self.connected = True
            self.last_error = None
            return True
        except Exception as exc:
            if self.connected:
                self._log("telemetry link down: %s" % exc)
            self.connected = False
            self.last_error = str(exc)[:200]
            return False

    # -- store and forward -------------------------------------------------

    def _write_offline(self, batch: list) -> None:
        try:
            with open(self.buffer_path, "a", encoding="utf-8") as handle:
                for record in batch:
                    handle.write(json.dumps(record) + "\n")
            self.buffered_count += len(batch)
            self._trim_offline_buffer()
        except Exception as exc:
            self._log("could not write offline buffer: %s" % exc)

    def _trim_offline_buffer(self) -> None:
        """Keep only the newest `buffer_max` records so the SD card cannot fill."""
        try:
            if not os.path.exists(self.buffer_path):
                return
            with open(self.buffer_path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            if len(lines) <= self.buffer_max:
                return
            with open(self.buffer_path, "w", encoding="utf-8") as handle:
                handle.writelines(lines[-self.buffer_max:])
        except Exception:
            pass

    def _flush_offline_buffer(self) -> None:
        """Replay buffered records, oldest first, once the link is back."""
        if not os.path.exists(self.buffer_path):
            return
        try:
            with open(self.buffer_path, "r", encoding="utf-8") as handle:
                lines = [ln for ln in handle if ln.strip()]
        except Exception:
            return
        if not lines:
            return

        self._log("replaying %d buffered records" % len(lines))
        remaining = list(lines)
        while remaining:
            chunk, remaining = remaining[:self.batch_max], remaining[self.batch_max:]
            try:
                records = [json.loads(ln) for ln in chunk]
            except Exception:
                continue                      # skip a corrupt chunk, keep going
            if not self._try_send(records):
                # Still down: put back what has not gone out and try later.
                try:
                    with open(self.buffer_path, "w", encoding="utf-8") as handle:
                        handle.writelines(chunk + remaining)
                except Exception:
                    pass
                return
            self.sent_count += len(records)

        try:
            os.remove(self.buffer_path)
            self.buffered_count = 0
        except Exception:
            pass

    # -- downlink ----------------------------------------------------------

    def _maybe_poll_commands(self) -> None:
        if time.time() - self._last_command_poll < 2.0:
            return
        self._last_command_poll = time.time()
        try:
            commands = self.transport.fetch_commands(self.device_id,
                                                     self._command_cursor)
        except Exception:
            return
        for command in commands:
            issued = float(command.get("issued_at", 0.0))
            self._command_cursor = max(self._command_cursor, issued)
            self._commands.put(command)
