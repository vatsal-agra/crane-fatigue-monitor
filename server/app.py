"""
Control-room server.

Provides three things:

  * an ingest endpoint the cabin units POST to (and replay into after an
    outage)
  * a live supervisor dashboard with a server-sent-events push channel
  * the post-shift report, which is the output that feeds back into rostering

Run it with:

    python -m server.app                 # http://127.0.0.1:5000
    python -m server.app --port 8080
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.config import load_config          # noqa: E402
from server.database import Database         # noqa: E402

VALID_COMMANDS = {"acknowledge", "recalibrate", "reset_counters",
                  "halt_crane", "pause_operation", "resume"}


def port_is_free(host: str, port: int) -> bool:
    """
    True if a server can bind this port right now.

    Deliberately does NOT set SO_REUSEADDR. On Windows that option permits
    binding a port another process is actively listening on, so the probe would
    report every port as free and the whole fallback would be silently useless.
    Without it, bind() fails on a port in use on every platform, which is
    exactly the question being asked.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def pick_port(host: str, preferred: int, attempts: int = 20) -> int:
    """
    Return the preferred port, or the next free one above it.

    This exists because of macOS specifically. Since Monterey, the AirPlay
    Receiver service listens on port 5000 by default - which is also Flask's
    default. On a Mac the server would either fail to bind or, worse, bind
    fine while the browser reaches AirPlay instead and shows nothing. Rather
    than make every Mac user hunt through System Settings, move to 5001.
    """
    for offset in range(attempts):
        candidate = preferred + offset
        if port_is_free(host, candidate):
            return candidate
    return preferred


class EventBus:
    """
    Fan-out for server-sent events.

    Each connected dashboard gets its own bounded queue. A browser tab that
    stops draining (minimised, throttled, or on a dead Wi-Fi link) must never
    be able to grow the server's memory without limit, so a full queue drops
    its oldest frame instead of blocking the ingest path.
    """

    def __init__(self, maxsize: int = 50):
        self._subscribers: list = []
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, payload: dict) -> None:
        message = json.dumps(payload)
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(message)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except (queue.Empty, queue.Full):
                    pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


def create_app(cfg=None) -> Flask:
    cfg = cfg or load_config()
    app = Flask(__name__)
    app.config["CFG"] = cfg

    db = Database(str(cfg.server.database))
    bus = EventBus()
    app.config["DB"] = db
    app.config["BUS"] = bus

    notify_states = set(cfg.server.get("notify_states", []))

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.route("/")
    def dashboard():
        return render_template("dashboard.html", cfg=cfg)

    @app.route("/report")
    def report_page():
        return render_template("report.html", cfg=cfg)

    # ------------------------------------------------------------------
    # Ingest - the cabin uplink
    # ------------------------------------------------------------------

    @app.post("/api/ingest")
    def ingest():
        payload = request.get_json(silent=True) or {}
        records = payload.get("records")
        if not isinstance(records, list):
            return jsonify({"error": "expected a 'records' array"}), 400

        written = db.ingest(records)

        # Push only what a dashboard actually needs to repaint, and only for
        # the newest record in the batch - a replayed backlog of 2 000 rows
        # must not become 2 000 SSE frames.
        if records:
            latest = max(records, key=lambda r: r.get("timestamp") or 0)
            bus.publish({"type": "telemetry", "record": _live_view(latest)})
            for record in records:
                for event in record.get("events") or []:
                    if event.get("severity") in ("warning", "critical"):
                        bus.publish({"type": "event",
                                     "event": event,
                                     "device_id": record.get("device_id"),
                                     "operator_id": record.get("operator_id")})
                if record.get("state") in notify_states and record.get("state_changed"):
                    bus.publish({"type": "alert",
                                 "device_id": record.get("device_id"),
                                 "state": record.get("state"),
                                 "score": record.get("score"),
                                 "operator_id": record.get("operator_id"),
                                 "timestamp": record.get("timestamp")})

        return jsonify({"ok": True, "written": written})

    def _live_view(record: dict) -> dict:
        keys = ("device_id", "crane_id", "site", "operator_id", "operator_name",
                "timestamp", "state", "score", "ear", "mar", "tilt", "heart_rate",
                "hrv_rmssd", "perclos", "counters", "sub_scores", "reasons",
                "face_visible", "vision_backend", "sensor_source", "calibrated")
        return {k: record.get(k) for k in keys}

    # ------------------------------------------------------------------
    # Live state
    # ------------------------------------------------------------------

    @app.get("/api/live")
    def live():
        return jsonify({
            "devices": db.devices(),
            "unacknowledged": db.unacknowledged_count(),
            "server_time": time.time(),
        })

    @app.get("/api/events")
    def events():
        return jsonify({"events": db.events(
            device_id=request.args.get("device_id"),
            limit=int(request.args.get("limit", 60)),
            severity=request.args.get("severity"),
            since=float(request.args["since"]) if request.args.get("since") else None,
            only_unacknowledged=request.args.get("unacknowledged") == "1",
        )})

    @app.post("/api/events/<int:event_id>/acknowledge")
    def acknowledge(event_id: int):
        by = (request.get_json(silent=True) or {}).get("by", "supervisor")
        ok = db.acknowledge_event(event_id, by)
        if ok:
            bus.publish({"type": "acknowledged", "event_id": event_id})
        return jsonify({"ok": ok})

    @app.get("/api/timeseries")
    def timeseries():
        device_id = request.args.get("device_id")
        if not device_id:
            return jsonify({"error": "device_id is required"}), 400
        return jsonify({"points": db.timeseries(
            device_id, minutes=float(request.args.get("minutes", 10)))})

    @app.get("/api/report")
    def report():
        return jsonify(db.shift_report(
            device_id=request.args.get("device_id") or None,
            hours=float(request.args.get("hours", 8))))

    # ------------------------------------------------------------------
    # Supervisor downlink
    # ------------------------------------------------------------------

    @app.get("/api/commands")
    def get_commands():
        """Polled by the edge node."""
        device_id = request.args.get("device_id")
        if not device_id:
            return jsonify({"error": "device_id is required"}), 400
        since = float(request.args.get("since", 0.0))
        return jsonify({"commands": db.pending_commands(device_id, since)})

    @app.post("/api/commands")
    def post_command():
        """Issued by the supervisor from the dashboard."""
        payload = request.get_json(silent=True) or {}
        device_id = payload.get("device_id")
        action = str(payload.get("action", "")).lower()

        if not device_id:
            return jsonify({"error": "device_id is required"}), 400
        if action not in VALID_COMMANDS:
            return jsonify({"error": "unknown action %r" % action,
                            "valid": sorted(VALID_COMMANDS)}), 400

        command_id = db.issue_command(device_id, action,
                                      note=str(payload.get("note", "")),
                                      issued_by=str(payload.get("by", "supervisor")))
        bus.publish({"type": "command", "device_id": device_id, "action": action})
        return jsonify({"ok": True, "command_id": command_id})

    @app.get("/api/command_history")
    def command_history():
        return jsonify({"commands": db.command_history()})

    # ------------------------------------------------------------------
    # Server-sent events
    # ------------------------------------------------------------------

    @app.get("/api/stream")
    def stream():
        def generate():
            q = bus.subscribe()
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        message = q.get(timeout=15.0)
                        yield "data: %s\n\n" % message
                    except queue.Empty:
                        yield ": keep-alive\n\n"   # stop proxies timing us out
            finally:
                bus.unsubscribe(q)

        return Response(generate(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    @app.get("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "database": str(cfg.server.database),
            "devices": len(db.devices()),
            "dashboards_connected": bus.subscriber_count,
            "server_time": time.time(),
        })

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Crane fatigue control-room server")
    parser.add_argument("--config")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--purge-days", type=int,
                        help="delete records older than N days and exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    host = args.host or str(cfg.server.host)
    port = args.port or int(cfg.server.port)

    if args.purge_days is not None:
        removed = Database(str(cfg.server.database)).purge_older_than(args.purge_days)
        print("purged %d telemetry rows older than %d days" % (removed, args.purge_days))
        return 0

    requested = port
    port = pick_port(host, port)

    app = create_app(cfg)
    print("=" * 68)
    print("  Crane Operator Fatigue Monitor - control room")
    print("=" * 68)
    if port != requested:
        print("  NOTE: port %d was busy, using %d instead." % (requested, port))
        if sys.platform == "darwin" and requested == 5000:
            print("        On macOS, port 5000 is usually the AirPlay Receiver.")
            print("        Turn it off in System Settings > General > AirDrop")
            print("        & Handoff if you want the default port back.")
        print("        Point cabin units at this server with:")
        print("          python -m edge.node --server http://%s:%d" % (host, port))
        print("-" * 68)
    print("  dashboard : http://%s:%d/" % (host, port))
    print("  report    : http://%s:%d/report" % (host, port))
    print("  database  : %s" % cfg.server.database)
    print("=" * 68)

    # threaded=True is required: the SSE endpoint holds a worker for the life
    # of each dashboard, and ingest must not queue behind it.
    app.run(host=host, port=port, debug=args.debug, threaded=True,
            use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
