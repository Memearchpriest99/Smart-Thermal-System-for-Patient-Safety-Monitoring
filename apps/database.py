"""
Ward Watcher — SQLite database layer.

Two tables:
  alerts      — every incoming alert, including dismiss state
  dismiss_log — audit trail of every dismiss action
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

# When frozen by PyInstaller, store the DB next to the exe so it survives updates.
# During development, store it next to this source file.
if getattr(sys, "frozen", False):
    _BASE = Path(sys.executable).parent
else:
    _BASE = Path(__file__).parent

DB_PATH = _BASE / "ward_watcher.db"

VALID_ALERT_TYPES = frozenset(
    {"fire", "unauthorized_presence", "unauthorized_touch"}
)


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """Open a SQLite connection, commit on success, rollback on error."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ------------------------------------------------------------------ #
# Schema                                                               #
# ------------------------------------------------------------------ #

def init_db() -> None:
    """Create tables if they do not already exist."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type   TEXT    NOT NULL,
                room_id      INTEGER NOT NULL,
                timestamp    TEXT    NOT NULL,
                dismissed    INTEGER NOT NULL DEFAULT 0,
                dismissed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS dismiss_log (
                log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id     INTEGER NOT NULL,
                alert_type   TEXT    NOT NULL,
                room_id      INTEGER NOT NULL,
                dismissed_at TEXT    NOT NULL
            );
        """)


# ------------------------------------------------------------------ #
# Alerts CRUD                                                          #
# ------------------------------------------------------------------ #

def insert_alert(alert_type: str, room_id: int, timestamp: str) -> int:
    """Insert a new alert and return its generated alert_id."""
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO alerts (alert_type, room_id, timestamp) VALUES (?, ?, ?)",
            (alert_type, int(room_id), timestamp),
        )
        return cursor.lastrowid


def get_all_alerts() -> list[dict]:
    """Return all alerts, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY alert_id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_alert(alert_id: int) -> Optional[dict]:
    """Return a single alert by ID, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        return dict(row) if row else None


def update_alert(alert_id: int, fields: dict) -> bool:
    """
    Update allowed fields on an alert row.

    Accepted keys: alert_type, room_id, timestamp, dismissed, dismissed_at.
    Returns False if no valid fields are supplied.
    """
    allowed = {"alert_type", "room_id", "timestamp", "dismissed", "dismissed_at"}
    clean = {k: v for k, v in fields.items() if k in allowed}
    if not clean:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in clean)
    values = list(clean.values()) + [alert_id]
    with _connect() as conn:
        conn.execute(f"UPDATE alerts SET {set_clause} WHERE alert_id = ?", values)
    return True


def delete_alert(alert_id: int) -> None:
    """Delete an alert row (and any associated dismiss log rows)."""
    with _connect() as conn:
        conn.execute("DELETE FROM dismiss_log WHERE alert_id = ?", (alert_id,))
        conn.execute("DELETE FROM alerts WHERE alert_id = ?", (alert_id,))


# ------------------------------------------------------------------ #
# Dismiss                                                              #
# ------------------------------------------------------------------ #

def dismiss_alert(alert_id: int, dismissed_at: str) -> bool:
    """
    Mark an alert as dismissed and write an entry to dismiss_log.

    Returns False if the alert does not exist or is already dismissed.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        if not row or row["dismissed"]:
            return False
        conn.execute(
            "UPDATE alerts SET dismissed = 1, dismissed_at = ? WHERE alert_id = ?",
            (dismissed_at, alert_id),
        )
        conn.execute(
            """INSERT INTO dismiss_log (alert_id, alert_type, room_id, dismissed_at)
               VALUES (?, ?, ?, ?)""",
            (alert_id, row["alert_type"], row["room_id"], dismissed_at),
        )
    return True


# ------------------------------------------------------------------ #
# Dismiss log                                                          #
# ------------------------------------------------------------------ #

def get_all_dismiss_logs() -> list[dict]:
    """Return all dismiss log entries, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM dismiss_log ORDER BY log_id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_dismiss_log(log_id: int) -> None:
    """Delete a single dismiss log entry."""
    with _connect() as conn:
        conn.execute("DELETE FROM dismiss_log WHERE log_id = ?", (log_id,))
