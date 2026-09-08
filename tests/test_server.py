"""
Tests for the control-room server: ingest, live view, alerting, the supervisor
downlink and the shift report.

Each test gets its own throwaway SQLite file, so nothing here touches the real
shift database.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.config import load_config
from server.app import create_app
from server.database import Database


def record(device_id="CRANE-01-CABIN", ts=None, state="Normal", score=12.0,
           events=None, operator_id="OP-1043", **extra):
    payload = {
        "device_id": device_id,
        "crane_id": "TOWER-CRANE-01",
        "site": "Site A",
        "operator_id": operator_id,
        "operator_name": "R. Kumar",
        "timestamp": ts if ts is not None else time.time(),
        "state": state,
        "score": score,
        "raw_score": score,
        "ear": 0.28, "mar": 0.15, "tilt": 3.0,
        "heart_rate": 72.0, "hrv_rmssd": 38.0, "perclos": 0.05,
        "counters": {"drowsiness": 0, "yawn": 0, "posture": 0, "vitals": 0},
        "sub_scores": {"drowsiness": 0.1, "yawn": 0.0, "posture": 0.0, "vitals": 0.0},
        "reasons": [], "face_visible": True,
        "vision_backend": "mediapipe", "sensor_source": "simulated",
        "calibrated": True, "events": events or [],
    }
    payload.update(extra)
    return payload


class ServerTestCase(unittest.TestCase):

    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        cfg = load_config()
        cfg.set_path("server.database", self.db_path)
        self.cfg = cfg
        self.app = create_app(cfg)
        self.client = self.app.test_client()

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(self.db_path + suffix)
            except OSError:
                pass

    def post(self, *records):
        return self.client.post("/api/ingest", json={"records": list(records)})


class TestIngest(ServerTestCase):

    def test_stores_telemetry_and_registers_the_device(self):
        response = self.post(record())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["written"], 1)

        devices = self.client.get("/api/live").get_json()["devices"]
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["device_id"], "CRANE-01-CABIN")
        self.assertTrue(devices[0]["online"])

    def test_rejects_a_malformed_body(self):
        self.assertEqual(self.client.post("/api/ingest", json={}).status_code, 400)

    def test_a_record_without_a_device_id_is_skipped_not_fatal(self):
        payload = record()
        del payload["device_id"]
        response = self.post(payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["written"], 0)

    def test_events_are_stored_and_escalated_ones_are_listed(self):
        self.post(record(state="Critical", score=82.0, events=[
            {"kind": "microsleep", "severity": "warning",
             "message": "Eyes closed", "value": 0.08, "timestamp": time.time()},
            {"kind": "yawn", "severity": "info",
             "message": "Yawn", "value": 0.7, "timestamp": time.time()},
        ]))
        everything = self.client.get("/api/events").get_json()["events"]
        escalated = self.client.get("/api/events?severity=escalated").get_json()["events"]
        self.assertEqual(len(everything), 2)
        self.assertEqual(len(escalated), 1)
        self.assertEqual(escalated[0]["kind"], "microsleep")

    def test_a_replayed_backlog_cannot_rewind_the_live_state(self):
        """
        The failure this guards against:

        a cabin unit loses its radio link, buffers an hour of history, and
        replays it after reconnecting. Those records are *older* than the live
        state the control room already holds. If the replay overwrote it, a
        crane currently in Critical would silently revert to whatever it was
        reporting an hour ago - on the supervisor's screen, in the middle of an
        active alert.
        """
        now = time.time()
        self.post(record(ts=now, state="Critical", score=88.0))
        self.post(record(ts=now - 3600, state="Normal", score=5.0))  # the replay

        device = self.client.get("/api/live").get_json()["devices"][0]
        self.assertEqual(device["last_state"], "Critical")
        self.assertAlmostEqual(device["last_score"], 88.0, places=1)

        # The replayed sample is still stored as history, just not as "latest".
        report = self.client.get("/api/report?hours=2").get_json()
        self.assertEqual(report["samples"], 2)

    def test_live_view_carries_the_full_latest_record(self):
        """The dashboard must paint a complete panel on first load."""
        self.post(record(score=44.0, state="Warning"))
        device = self.client.get("/api/live").get_json()["devices"][0]
        self.assertIsNotNone(device["latest"])
        self.assertAlmostEqual(device["latest"]["score"], 44.0, places=1)
        self.assertIn("counters", device["latest"])
        self.assertIn("sub_scores", device["latest"])


class TestStaleDevices(ServerTestCase):

    def test_a_silent_unit_is_marked_offline_with_no_stale_score(self):
        self.post(record(ts=time.time() - 600, state="Critical", score=91.0))
        device = self.client.get("/api/live").get_json()["devices"][0]
        self.assertFalse(device["online"])
        self.assertEqual(device["last_state"], "Offline")
        # A stale 91 beside an OFFLINE badge would read as a live measurement.
        self.assertIsNone(device["last_score"])
        self.assertIsNone(device["latest"]["score"])
        self.assertTrue(device["latest"]["stale"])


class TestAcknowledgement(ServerTestCase):

    def test_an_alert_can_be_acknowledged_once(self):
        self.post(record(events=[{"kind": "microsleep", "severity": "critical",
                                  "message": "Eyes closed", "value": 0.07,
                                  "timestamp": time.time()}]))
        event_id = self.client.get("/api/events").get_json()["events"][0]["id"]
        self.assertEqual(self.client.get("/api/live").get_json()["unacknowledged"], 1)

        first = self.client.post("/api/events/%d/acknowledge" % event_id,
                                 json={"by": "supervisor"})
        self.assertTrue(first.get_json()["ok"])
        # Acknowledging twice must not double-count or error.
        second = self.client.post("/api/events/%d/acknowledge" % event_id, json={})
        self.assertFalse(second.get_json()["ok"])
        self.assertEqual(self.client.get("/api/live").get_json()["unacknowledged"], 0)


class TestCommands(ServerTestCase):

    def test_a_command_is_delivered_exactly_once(self):
        self.post(record())
        issued = self.client.post("/api/commands", json={
            "device_id": "CRANE-01-CABIN", "action": "halt_crane"})
        self.assertTrue(issued.get_json()["ok"])

        first = self.client.get(
            "/api/commands?device_id=CRANE-01-CABIN&since=0").get_json()["commands"]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["action"], "halt_crane")

        # Once collected by the cabin it must not be delivered again, or a
        # halt would be re-applied on every poll.
        second = self.client.get(
            "/api/commands?device_id=CRANE-01-CABIN&since=0").get_json()["commands"]
        self.assertEqual(len(second), 0)

    def test_unknown_actions_are_refused(self):
        response = self.client.post("/api/commands", json={
            "device_id": "CRANE-01-CABIN", "action": "launch_the_boom"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid", response.get_json())

    def test_a_command_needs_a_device(self):
        self.assertEqual(self.client.post(
            "/api/commands", json={"action": "resume"}).status_code, 400)


class TestReport(ServerTestCase):

    def test_summarises_the_window(self):
        now = time.time()
        for i in range(60):
            self.post(record(ts=now - (60 - i) * 60,
                             state="Warning" if i % 2 else "Normal",
                             score=20.0 + i))
        report = self.client.get("/api/report?hours=2").get_json()
        self.assertEqual(report["samples"], 60)
        self.assertAlmostEqual(report["peak_score"], 79.0, places=1)
        self.assertIn("Warning", report["state_distribution"])
        self.assertTrue(report["hourly"])

    def test_can_be_filtered_to_one_cabin(self):
        self.post(record(device_id="CRANE-01-CABIN", score=10.0))
        self.post(record(device_id="CRANE-02-CABIN", score=90.0,
                         operator_id="OP-2210"))
        one = self.client.get(
            "/api/report?hours=2&device_id=CRANE-02-CABIN").get_json()
        self.assertEqual(one["samples"], 1)
        self.assertAlmostEqual(one["peak_score"], 90.0, places=1)

    def test_empty_database_reports_zeroes_rather_than_failing(self):
        report = self.client.get("/api/report?hours=8").get_json()
        self.assertEqual(report["samples"], 0)
        self.assertEqual(report["mean_score"], 0.0)
        self.assertEqual(report["total_alerts"], 0)


class TestTimeseries(ServerTestCase):

    def test_requires_a_device(self):
        self.assertEqual(self.client.get("/api/timeseries").status_code, 400)

    def test_decimates_long_windows(self):
        """A whole shift must not send 100 000 rows to draw 600 pixels."""
        now = time.time()
        self.post(*[record(ts=now - i) for i in range(3000, 0, -1)])
        points = self.client.get(
            "/api/timeseries?device_id=CRANE-01-CABIN&minutes=60"
        ).get_json()["points"]
        self.assertLessEqual(len(points), 700)
        self.assertGreater(len(points), 100)
        # Still in chronological order after decimation.
        stamps = [p["ts"] for p in points]
        self.assertEqual(stamps, sorted(stamps))


class TestMigration(unittest.TestCase):

    def test_a_column_added_later_is_applied_to_an_existing_database(self):
        """Shift databases hold real logged history and must not be recreated."""
        handle, path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        try:
            import sqlite3
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE devices (device_id TEXT PRIMARY KEY, "
                         "last_seen REAL)")
            conn.execute("INSERT INTO devices VALUES ('OLD-UNIT', 1.0)")
            conn.commit()
            conn.close()

            db = Database(path)
            columns = {row["name"] for row in
                       db.connect().execute("PRAGMA table_info(devices)")}
            self.assertIn("latest_json", columns)
            kept = db.connect().execute(
                "SELECT COUNT(*) AS n FROM devices").fetchone()["n"]
            self.assertEqual(kept, 1, "existing history must survive migration")
        finally:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(path + suffix)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
