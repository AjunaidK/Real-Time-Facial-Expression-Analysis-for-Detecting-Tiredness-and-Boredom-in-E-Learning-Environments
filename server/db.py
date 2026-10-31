"""
Minimal SQLite store for evaluation measurements.

Records ONLY derived numbers, timing, and (for the volunteer study) consent /
self-report. No image is ever written — this is the implementation-level
expression of the privacy-by-design commitment (Chapter 4.8).
"""

from __future__ import annotations
import sqlite3
import time
import threading


SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    session_id   TEXT NOT NULL,
    tiredness    REAL,
    boredom      REAL,
    face_present INTEGER,
    server_ms    REAL,
    self_report  TEXT
);
CREATE TABLE IF NOT EXISTS consent (
    session_id   TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    consented    INTEGER NOT NULL,
    note         TEXT
);
"""


class Database:
    def __init__(self, path: str = "eval.db"):
        self.path = path
        self._lock = threading.Lock()
        con = self._connect()
        con.executescript(SCHEMA)
        con.commit()
        con.close()

    def _connect(self):
        return sqlite3.connect(self.path)

    def log(self, session_id, tiredness, boredom, face_present,
            server_ms, self_report=None):
        with self._lock:
            con = self._connect()
            con.execute(
                "INSERT INTO measurements "
                "(ts, session_id, tiredness, boredom, face_present, "
                " server_ms, self_report) VALUES (?,?,?,?,?,?,?)",
                (time.time(), session_id, tiredness, boredom,
                 int(bool(face_present)), server_ms, self_report),
            )
            con.commit()
            con.close()

    def record_consent(self, session_id, consented, note=None):
        with self._lock:
            con = self._connect()
            con.execute(
                "INSERT OR REPLACE INTO consent (session_id, ts, consented, note)"
                " VALUES (?,?,?,?)",
                (session_id, time.time(), int(bool(consented)), note),
            )
            con.commit()
            con.close()
