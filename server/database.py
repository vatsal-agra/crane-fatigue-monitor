"""
Control-room datastore.

SQLite is deliberate rather than a compromise: the whole control-room service
is meant to run on a single site PC or on the site gateway itself, it needs no
administration, and the entire shift history is one file a supervisor can copy
onto a USB stick. Swapping in MySQL later means changing this module only -
nothing above it writes SQL.

Schema
    devices    one row per cabin unit, holding its latest known state
    telemetry  the time series behind the charts and the shift report
    events     discrete alerts (microsleep, yawn, nod-off, state changes)
    commands   supervisor -> cabin downlink, polled by the edge node
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id       TEXT PRIMARY KEY,
    crane_id        TEXT,
    site            TEXT,
    operator_id     TEXT,
    operator_name   TEXT,
    last_seen       REAL,
    last_state      TEXT,
    last_score      REAL,
    vision_backend  TEXT,
    sensor_source   TEXT,
    calibrated      INTEGER DEFAULT 0,
    -- The full most recent telemetry record, verbatim. Keeping it here means
    -- /api/live can render a complete operator panel on first paint instead of
    -- waiting for the next push, and the view survives a server restart.
    latest_json     TEXT
);

CREATE TABLE IF NOT EXISTS telemetry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT NOT NULL,
    operator_id   TEXT,
    ts            REAL NOT NULL,
    state         TEXT,
    score         REAL,
    raw_score     REAL,
    ear           REAL,
    mar           REAL,
    tilt          REAL,
    heart_rate    REAL,
    hrv_rmssd     REAL,
    perclos       REAL,
    c_drowsiness  INTEGER,
    c_yawn        INTEGER,
    c_posture     INTEGER,
    c_vitals      INTEGER,
    face_visible  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_telemetry_device_ts ON telemetry(device_id, ts);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id       TEXT NOT NULL,
    operator_id     TEXT,
    ts              REAL NOT NULL,
    kind            TEXT,
    severity        TEXT,
    message         TEXT,
    value           REAL,
    score           REAL,
    acknowledged    INTEGER DEFAULT 0,
    acknowledged_at REAL,
    acknowledged_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_device ON events(device_id, ts);

CREATE TABLE IF NOT EXISTS commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    note        TEXT,
    issued_at   REAL NOT NULL,
    issued_by   TEXT,
    delivered   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_commands_device ON commands(device_id, issued_at);
"""

# Events worth surfacing on the dashboard alert feed; routine 'info' chatter
# such as an ordinary yawn is stored but not escalated.
ESCALATED_SEVERITIES = ("warning", "critical")

# Fields kept in devices.latest_json - everything the live operator panel
# renders, and nothing else. The per-frame event list is excluded because
# events have their own table and their own feed.
LIVE_FIELDS = ("device_id", "crane_id", "site", "operator_id", "operator_name",
               "timestamp", "state", "score", "raw_score", "ear", "mar", "tilt",
               "heart_rate", "hrv_rmssd", "perclos", "counters", "sub_scores",
               "reasons", "face_visible", "vision_backend", "sensor_source",
               "calibrated", "ear_threshold", "mar_threshold")


def _live_fields(record: dict) -> dict:
    return {k: record.get(k) for k in LIVE_FIELDS}


class Database:
    """Thread-safe SQLite wrapper - one connection per thread."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # WAL lets the dashboard read while the ingest endpoint writes.
            conn.execute("PRAGMA journal_mode=WAL")
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """
        Additive migrations for databases created by an earlier version.

        CREATE TABLE IF NOT EXISTS silently does nothing when the table already
        exists, so a column added later never appears in an existing shift
        database. Since these files hold real logged history that must not be
        thrown away, missing columns are added in place.
        """
        expected = {
            "devices": {"latest_json": "TEXT"},
        }
        for table, columns in expected.items():
            existing = {row["name"] for row in
                        conn.execute("PRAGMA table_info(%s)" % table).fetchall()}
            for name, decl in columns.items():
                if name not in existing:
                    conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, name, decl))

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def ingest(self, records: list) -> int:
        """
        Store a batch of telemetry records, and any events they carry.

        Returns the number of telemetry rows written. Records replayed from an
        edge node's offline buffer arrive here exactly like live ones, so a
        reconnect backfills the history automatically.
        """
        written = 0
        conn = self.connect()
        with self._write_lock, conn:
            for record in records:
                device_id = record.get("device_id")
                if not device_id:
                    continue
                counters = record.get("counters") or {}
                ts = float(record.get("timestamp") or time.time())

                conn.execute(
                    """INSERT INTO telemetry
                       (device_id, operator_id, ts, state, score, raw_score, ear, mar,
                        tilt, heart_rate, hrv_rmssd, perclos, c_drowsiness, c_yawn,
                        c_posture, c_vitals, face_visible)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (device_id, record.get("operator_id"), ts, record.get("state"),
                     record.get("score"), record.get("raw_score"), record.get("ear"),
                     record.get("mar"), record.get("tilt"), record.get("heart_rate"),
                     record.get("hrv_rmssd"), record.get("perclos"),
                     counters.get("drowsiness"), counters.get("yawn"),
                     counters.get("posture"), counters.get("vitals"),
                     1 if record.get("face_visible") else 0))
                written += 1

                # A record replayed from an edge node's offline buffer is older
                # than what the device row already holds. Every "latest" field
                # is therefore guarded: only a record at least as new as the
                # stored one may overwrite the current view of the cabin.
                # Without this, a reconnecting node would roll a live Critical
                # state back to whatever it was reporting an hour ago.
                conn.execute(
                    """INSERT INTO devices (device_id, crane_id, site, operator_id,
                            operator_name, last_seen, last_state, last_score,
                            vision_backend, sensor_source, calibrated, latest_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(device_id) DO UPDATE SET
                            crane_id=excluded.crane_id,
                            site=excluded.site,
                            last_state=CASE WHEN excluded.last_seen >= devices.last_seen
                                       THEN excluded.last_state ELSE devices.last_state END,
                            last_score=CASE WHEN excluded.last_seen >= devices.last_seen
                                       THEN excluded.last_score ELSE devices.last_score END,
                            operator_id=CASE WHEN excluded.last_seen >= devices.last_seen
                                       THEN excluded.operator_id ELSE devices.operator_id END,
                            operator_name=CASE WHEN excluded.last_seen >= devices.last_seen
                                       THEN excluded.operator_name ELSE devices.operator_name END,
                            vision_backend=CASE WHEN excluded.last_seen >= devices.last_seen
                                       THEN excluded.vision_backend ELSE devices.vision_backend END,
                            sensor_source=CASE WHEN excluded.last_seen >= devices.last_seen
                                       THEN excluded.sensor_source ELSE devices.sensor_source END,
                            calibrated=CASE WHEN excluded.last_seen >= devices.last_seen
                                       THEN excluded.calibrated ELSE devices.calibrated END,
                            latest_json=CASE WHEN excluded.last_seen >= devices.last_seen
                                       THEN excluded.latest_json ELSE devices.latest_json END,
                            last_seen=MAX(devices.last_seen, excluded.last_seen)""",
                    (device_id, record.get("crane_id"), record.get("site"),
                     record.get("operator_id"), record.get("operator_name"), ts,
                     record.get("state"), record.get("score"),
                     record.get("vision_backend"), record.get("sensor_source"),
                     1 if record.get("calibrated") else 0,
                     json.dumps(_live_fields(record))))

                for event in record.get("events") or []:
                    conn.execute(
                        """INSERT INTO events (device_id, operator_id, ts, kind,
                                severity, message, value, score)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (device_id, record.get("operator_id"),
                         float(event.get("timestamp") or ts), event.get("kind"),
                         event.get("severity"), event.get("message"),
                         event.get("value"), record.get("score")))
        return written

    def issue_command(self, device_id: str, action: str,
                      note: str = "", issued_by: str = "supervisor") -> int:
        conn = self.connect()
        with self._write_lock, conn:
            cur = conn.execute(
                """INSERT INTO commands (device_id, action, note, issued_at, issued_by)
                   VALUES (?,?,?,?,?)""",
                (device_id, action, note, time.time(), issued_by))
            return int(cur.lastrowid)

    def acknowledge_event(self, event_id: int, by: str = "supervisor") -> bool:
        conn = self.connect()
        with self._write_lock, conn:
            cur = conn.execute(
                """UPDATE events SET acknowledged=1, acknowledged_at=?, acknowledged_by=?
                   WHERE id=? AND acknowledged=0""",
                (time.time(), by, event_id))
            return cur.rowcount > 0

    def purge_older_than(self, days: int) -> int:
        cutoff = time.time() - days * 86400
        conn = self.connect()
        with self._write_lock, conn:
            n = conn.execute("DELETE FROM telemetry WHERE ts < ?", (cutoff,)).rowcount
            conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            conn.execute("DELETE FROM commands WHERE issued_at < ?", (cutoff,))
        return n

    def reset(self) -> None:
        """Wipe all operational data - used by the demo data loader."""
        conn = self.connect()
        with self._write_lock, conn:
            for table in ("telemetry", "events", "commands", "devices"):
                conn.execute("DELETE FROM %s" % table)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def devices(self, stale_after: float = 10.0) -> list:
        """Every known cabin unit with its latest state and liveness flag."""
        now = time.time()
        rows = self.connect().execute(
            "SELECT * FROM devices ORDER BY device_id").fetchall()
        out = []
        for row in rows:
            item = dict(row)
            last_seen = item.get("last_seen") or 0.0
            item["online"] = (now - last_seen) < stale_after
            item["seconds_since_seen"] = round(now - last_seen, 1)
            latest = None
            raw = item.pop("latest_json", None)
            if raw:
                try:
                    latest = json.loads(raw)
                except (TypeError, ValueError):
                    latest = None

            if not item["online"]:
                # A unit that has stopped reporting is itself a safety concern.
                # Neither its last state nor its last score may keep being shown
                # as though it were current - a stale "62" beside an OFFLINE
                # badge reads as a live measurement.
                item["last_state"] = "Offline"
                item["last_score"] = None
                if latest is not None:
                    latest = dict(latest, state="Offline", score=None, stale=True)

            item["latest"] = latest
            out.append(item)
        return out

    def unacknowledged_count(self) -> int:
        row = self.connect().execute(
            "SELECT COUNT(*) AS n FROM events WHERE acknowledged=0 AND severity IN (?,?)",
            ESCALATED_SEVERITIES).fetchone()
        return int(row["n"])

    def events(self, device_id: Optional[str] = None, limit: int = 60,
               severity: Optional[str] = None, since: Optional[float] = None,
               only_unacknowledged: bool = False) -> list:
        sql = "SELECT * FROM events WHERE 1=1"
        params: list = []
        if device_id:
            sql += " AND device_id = ?"
            params.append(device_id)
        if severity == "escalated":
            sql += " AND severity IN (?,?)"
            params.extend(ESCALATED_SEVERITIES)
        elif severity:
            sql += " AND severity = ?"
            params.append(severity)
        if since is not None:
            sql += " AND ts > ?"
            params.append(since)
        if only_unacknowledged:
            sql += " AND acknowledged = 0"
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self.connect().execute(sql, params).fetchall()]

    def timeseries(self, device_id: str, minutes: float = 10.0,
                   max_points: int = 600) -> list:
        """
        Recent telemetry for the charts.

        Long windows are decimated in SQL rather than in the browser, so a
        supervisor asking for a whole shift does not pull 100 000 rows over the
        network to draw 600 pixels.
        """
        since = time.time() - minutes * 60.0
        conn = self.connect()
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM telemetry WHERE device_id=? AND ts>?",
            (device_id, since)).fetchone()["n"]
        stride = max(1, int(total // max_points) + (1 if total > max_points else 0))

        rows = conn.execute(
            """SELECT ts, score, state, ear, mar, heart_rate, perclos, tilt, face_visible
               FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY ts) AS rn
                     FROM telemetry WHERE device_id=? AND ts>?)
               WHERE rn % ? = 0 ORDER BY ts""",
            (device_id, since, stride)).fetchall()
        return [dict(r) for r in rows]

    def shift_report(self, device_id: Optional[str] = None,
                     hours: float = 8.0) -> dict:
        """
        Step 16 of the algorithm: the post-shift summary.

        Returns the fatigue trend, alert counts by type, the state distribution
        and the fatigue-prone hours that feed back into shift scheduling.
        """
        since = time.time() - hours * 3600.0
        conn = self.connect()

        where = "ts > ?"
        params: list = [since]
        if device_id:
            where += " AND device_id = ?"
            params.append(device_id)

        summary = conn.execute(
            """SELECT COUNT(*) AS samples, AVG(score) AS mean_score,
                      MAX(score) AS peak_score, MIN(ts) AS first_ts, MAX(ts) AS last_ts
               FROM telemetry WHERE %s""" % where, params).fetchone()

        states = conn.execute(
            """SELECT state, COUNT(*) AS n FROM telemetry WHERE %s
               GROUP BY state""" % where, params).fetchall()

        kinds = conn.execute(
            """SELECT kind, severity, COUNT(*) AS n FROM events WHERE %s
               GROUP BY kind, severity ORDER BY n DESC""" % where, params).fetchall()

        # Fatigue by clock hour - the output that actually changes rosters.
        hourly = conn.execute(
            """SELECT CAST(strftime('%%H', ts, 'unixepoch', 'localtime') AS INTEGER) AS hour,
                      AVG(score) AS mean_score, MAX(score) AS peak_score, COUNT(*) AS n
               FROM telemetry WHERE %s
               GROUP BY hour ORDER BY hour""" % where, params).fetchall()

        peak = conn.execute(
            """SELECT ts, score, state, device_id, operator_id FROM telemetry
               WHERE %s ORDER BY score DESC LIMIT 1""" % where, params).fetchone()

        operators = conn.execute(
            """SELECT operator_id, COUNT(*) AS samples, AVG(score) AS mean_score,
                      MAX(score) AS peak_score
               FROM telemetry WHERE %s AND operator_id IS NOT NULL
               GROUP BY operator_id ORDER BY mean_score DESC""" % where,
            params).fetchall()

        total_states = sum(r["n"] for r in states) or 1
        return {
            "window_hours": hours,
            "device_id": device_id,
            "samples": summary["samples"] or 0,
            "mean_score": round(summary["mean_score"], 1) if summary["mean_score"] else 0.0,
            "peak_score": round(summary["peak_score"], 1) if summary["peak_score"] else 0.0,
            "first_ts": summary["first_ts"],
            "last_ts": summary["last_ts"],
            "state_distribution": {
                r["state"]: round(100.0 * r["n"] / total_states, 1) for r in states},
            "event_counts": [dict(r) for r in kinds],
            "hourly": [{"hour": r["hour"],
                        "mean_score": round(r["mean_score"] or 0, 1),
                        "peak_score": round(r["peak_score"] or 0, 1),
                        "samples": r["n"]} for r in hourly],
            "peak_moment": dict(peak) if peak else None,
            "operators": [{"operator_id": r["operator_id"], "samples": r["samples"],
                           "mean_score": round(r["mean_score"] or 0, 1),
                           "peak_score": round(r["peak_score"] or 0, 1)}
                          for r in operators],
            "total_alerts": sum(r["n"] for r in kinds
                                if r["severity"] in ESCALATED_SEVERITIES),
        }

    def pending_commands(self, device_id: str, since: float) -> list:
        """Commands the cabin has not yet collected."""
        conn = self.connect()
        rows = conn.execute(
            """SELECT * FROM commands
               WHERE device_id=? AND delivered=0 AND issued_at > ?
               ORDER BY issued_at""", (device_id, since)).fetchall()
        if rows:
            with self._write_lock, conn:
                conn.executemany("UPDATE commands SET delivered=1 WHERE id=?",
                                 [(r["id"],) for r in rows])
        return [dict(r) for r in rows]

    def command_history(self, limit: int = 30) -> list:
        return [dict(r) for r in self.connect().execute(
            "SELECT * FROM commands ORDER BY issued_at DESC LIMIT ?",
            (int(limit),)).fetchall()]
